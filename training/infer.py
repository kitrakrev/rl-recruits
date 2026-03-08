"""
Inference script — run a trained LoRA adapter against the live environment.

Loads the saved LoRA adapter from a checkpoint directory, runs one full
episode, and prints every action + reward.  No training, no gradient tracking.

Usage
-----
# Against a running env server (start with uvicorn server.app:app ...):
uv run python -m training.infer \\
    --checkpoint training/checkpoints \\
    --env_url    http://localhost:8000 \\
    --seed       42

# Compare against base model (no adapter):
uv run python -m training.infer \\
    --model_name Qwen/Qwen3-0.6B \\
    --no_adapter \\
    --env_url    http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained LoRA adapter for one inference episode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   default="training/checkpoints",
                   help="Directory containing the saved LoRA adapter (from model.save_pretrained)")
    p.add_argument("--model_name",   default=None,
                   help="Base model HuggingFace ID. Auto-detected from checkpoint if omitted.")
    p.add_argument("--env_url",      default="http://localhost:8000",
                   help="Environment server URL")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--max_turns",    type=int, default=10,
                   help="Max tool calls per week before auto-advance")
    p.add_argument("--no_adapter",   action="store_true",
                   help="Load base model only (no LoRA); useful as a baseline comparison")
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_p",        type=float, default=0.8)
    p.add_argument("--top_k",        type=int,   default=20)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--save_plot", default=None, metavar="PATH",
                   help="Save per-step reward curve to PNG (e.g. training/infer_rewards.png)")
    p.add_argument("--wandb", action="store_true",
                   help="Log run to Weights & Biases (requires WANDB_API_KEY)")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name (default: Staffing_agent or WANDB_PROJECT env)")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (default: infer-{checkpoint|base}-{seed})")
    p.add_argument("--episode_weeks", type=int, default=None,
                   help="Max weeks per episode (patches env; default uses server config)")
    p.add_argument("--max_loss", type=float, default=None,
                   help="End episode when cash drops by this much (0=disable). Patches env.")
    return p.parse_args()


def _detect_base_model(checkpoint_dir: str) -> str | None:
    """Read base model name from adapter_config.json if present."""
    cfg_path = Path(checkpoint_dir) / "adapter_config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            return data.get("base_model_name_or_path")
        except Exception:
            pass
    return None


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: uv pip install transformers torch peft")
        sys.exit(1)

    # ── Resolve base model name ─────────────────────────────────────────────
    model_name = args.model_name
    if model_name is None and not args.no_adapter:
        model_name = _detect_base_model(args.checkpoint)
    if model_name is None:
        print("[!] Could not detect base model name. Pass --model_name explicitly.")
        sys.exit(1)

    # ── Verify env server ───────────────────────────────────────────────────
    try:
        import requests
        requests.get(f"{args.env_url.rstrip('/')}/health", timeout=5).raise_for_status()
        print(f"[✓] Env server reachable at {args.env_url}")
    except Exception as e:
        print(f"[✗] Env server not reachable: {e}")
        print(f"    Start it with: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # ── Patch env config for episode length / early-stop ────────────────────
    if args.episode_weeks is not None or args.max_loss is not None:
        try:
            import requests
            patches = {}
            if args.episode_weeks is not None:
                patches["episode_steps"] = args.episode_weeks
            if args.max_loss is not None:
                patches["max_cumulative_loss"] = args.max_loss
            resp = requests.patch(
                f"{args.env_url.rstrip('/')}/config/env",
                json=patches,
                timeout=5,
            )
            print(f"[config] Patched env: {patches}  → HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[!] Could not patch env config: {exc}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint if not args.no_adapter else model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.no_adapter:
        model = base_model
        print("[i] Running base model (no LoRA adapter)")
    else:
        from peft import PeftModel
        checkpoint = args.checkpoint
        print(f"Loading LoRA adapter from: {checkpoint}")
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model = model.merge_and_unload()   # merge weights for faster inference
        print("[✓] LoRA adapter merged into base model")

    model.eval()

    # ── Build a minimal TrainingConfig-like namespace for rollout ───────────
    from dataclasses import dataclass

    @dataclass
    class InferCfg:
        max_prompt_len: int     = args.max_prompt_len
        max_full_len: int       = args.max_prompt_len + args.max_new_tokens
        max_new_tokens: int     = args.max_new_tokens
        temperature: float      = args.temperature
        top_p: float            = args.top_p
        top_k: int              = args.top_k
        max_turns_per_step: int = args.max_turns

    cfg = InferCfg()

    # ── Optional WandB init ─────────────────────────────────────────────────
    wb_run = None
    if args.wandb:
        try:
            import wandb
            import os
            run_name = args.wandb_run_name or (
                f"infer-{'base' if args.no_adapter else Path(args.checkpoint).name}-s{args.seed}"
            )
            proj_str = args.wandb_project or os.getenv("WANDB_PROJECT", "Staffing_agent")
            parts = proj_str.split("/", 1)
            project = parts[-1]
            entity = parts[0] if len(parts) == 2 else None
            wb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "no_adapter": args.no_adapter,
                    "seed": args.seed,
                    "model_name": model_name,
                },
            )
            print(f"[✓] W&B run: {wb_run.url}")
        except ImportError:
            print("[!] wandb not installed — skipping. Run: uv pip install wandb")
        except Exception as e:
            print(f"[!] WandB init failed: {e}")

    # ── Run inference episode ───────────────────────────────────────────────
    from client import StaffingAgencyEnv
    from training.rollout import rollout_full_episode

    rng = random.Random(args.seed)

    print(f"\n{'='*60}")
    print(f"  INFERENCE EPISODE  seed={args.seed}  adapter={'none (base)' if args.no_adapter else args.checkpoint}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        with StaffingAgencyEnv(base_url=args.env_url) as env:
            prompts, completions, rewards, final_profit, total_reward = rollout_full_episode(
                env, model, tokenizer,
                seed=args.seed,
                max_turns_per_step=args.max_turns,
                rng=rng,
                cfg=cfg,
            )

    # Log per-step rewards to WandB so all charts show a time series (not just one point)
    if wb_run is not None and rewards:
        cumulative = 0.0
        running_min = float("inf")
        running_max = float("-inf")
        for step, r in enumerate(rewards):
            cumulative += r
            running_min = min(running_min, r)
            running_max = max(running_max, r)
            running_mean = cumulative / (step + 1)
            wb_run.log({
                "infer/step_reward":       r,
                "infer/cumulative":        cumulative,
                "infer/step_index":        step,
                "infer/total_reward":      cumulative,
                "infer/steps":             step + 1,
                "infer/step_reward_min":   running_min,
                "infer/step_reward_max":   running_max,
                "infer/step_reward_mean":  running_mean,
                "infer/reward_sum":        cumulative,
            }, step=step)

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"    Final profit : ${final_profit:>12,.2f}")
    print(f"    Total reward : {total_reward:>+12,.2f}")
    print(f"    Steps taken  : {len(rewards)}")
    print(f"    Reward sum   : {sum(rewards):>+12,.2f}")
    print(f"{'='*60}")

    # ── Save plot (per-step rewards over time) ──────────────────────────────
    if args.save_plot and rewards:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            x = np.arange(len(rewards))
            ax.bar(x, rewards, color=["#27ae60" if r >= 0 else "#e74c3c" for r in rewards], alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_title(f"Inference Episode — seed={args.seed}  total_reward={total_reward:+.2f}  profit=${final_profit:,.0f}")
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[✓] Saved reward plot: {args.save_plot}")
        except ImportError:
            print("[!] matplotlib not installed — skipping plot. Run: uv pip install matplotlib")

    # ── Log final profit (only metric not available per-step) ─────────────────
    if wb_run is not None:
        last_step = max(0, len(rewards) - 1) if rewards else 0
        wb_run.log({"infer/final_profit": final_profit}, step=last_step)
        wb_run.finish()


if __name__ == "__main__":
    main()
