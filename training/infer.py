"""
<<<<<<< HEAD
Staffing Agency — Inference Script with WandB Logging
======================================================

Run inference episodes (no training) and log metrics to Weights & Biases.

Usage:
    # Base model only (no LoRA)
    uv run python training/infer.py --no_adapter --num_episodes 5

    # With LoRA checkpoint
    uv run python training/infer.py --checkpoint training/checkpoints --num_episodes 10

    # With WandB logging
    uv run python training/infer.py --checkpoint training/checkpoints --wandb --wandb_project myorg/staffing-agent

Requires env server running:
    uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
=======
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
>>>>>>> origin/main
"""
from __future__ import annotations

import argparse
<<<<<<< HEAD
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference episodes and log to WandB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None,
                   help="Path to LoRA adapter (e.g. training/checkpoints). Omit for base model only.")
    p.add_argument("--no_adapter", action="store_true",
                   help="Use base model only, no LoRA (ignores --checkpoint)")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B",
                   help="Base model ID (used when --no_adapter or as base for --checkpoint)")
    p.add_argument("--env_url", default="http://localhost:8000",
                   help="Environment server URL")
    p.add_argument("--num_episodes", type=int, default=5,
                   help="Number of episodes to run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_turns_per_step", type=int, default=10,
                   help="Max tool calls per week before auto-advance")
    p.add_argument("--episode_weeks", type=int, default=None,
                   help="Override env episode_steps (PATCH /config/env)")
    p.add_argument("--wandb", action="store_true",
                   help="Enable WandB logging")
    p.add_argument("--wandb_project", default=None,
                   help="WandB project (e.g. myorg/staffing-agent)")
    p.add_argument("--wandb_run_name", default=None,
                   help="WandB run name")
    p.add_argument("--wandb_api_key", default=os.getenv("WANDB_API_KEY"))
    return p.parse_args()


=======
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


>>>>>>> origin/main
def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
<<<<<<< HEAD
        from peft import PeftModel
=======
>>>>>>> origin/main
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: uv pip install transformers torch peft")
        sys.exit(1)

<<<<<<< HEAD
    # Verify env server
    try:
        import requests
        requests.get(f"{args.env_url}/health", timeout=5).raise_for_status()
        print(f"[✓] Env server healthy at {args.env_url}")
    except Exception as e:
        print(f"[✗] Env server not reachable at {args.env_url}: {e}")
        print("    Start with: uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Optional: PATCH env config (episode length)
    if args.episode_weeks is not None:
        try:
            resp = requests.patch(
                f"{args.env_url.rstrip('/')}/config/env",
                json={"episode_steps": args.episode_weeks},
                timeout=5,
            )
            print(f"[config] Patched episode_steps={args.episode_weeks} → HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[config] Warning: could not patch env: {exc}")

    # Load model
    model_name = args.model_name
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
=======
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
>>>>>>> origin/main
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

<<<<<<< HEAD
    if not args.no_adapter and args.checkpoint:
        print(f"Loading LoRA adapter from {args.checkpoint}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        model = base_model
        print("Using base model only (no LoRA)")

    model.eval()

    # Build minimal config for rollout
    from training.config import TrainingConfig
    cfg = TrainingConfig(
        model_name=model_name,
        max_prompt_len=1024,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

    # WandB setup
=======
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
>>>>>>> origin/main
    wb_run = None
    if args.wandb:
        try:
            import wandb
<<<<<<< HEAD
            if args.wandb_api_key:
                wandb.login(key=args.wandb_api_key, relogin=True)
            default_project = TrainingConfig().wandb_project
            entity_project = (args.wandb_project or default_project).split("/", 1)
            entity = entity_project[0] if len(entity_project) > 1 else None
            project = entity_project[-1]
            wb_run = wandb.init(
                entity=entity,
                project=project,
                name=args.wandb_run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "no_adapter": args.no_adapter,
                    "model_name": model_name,
                    "env_url": args.env_url,
                    "num_episodes": args.num_episodes,
                    "seed": args.seed,
                    "episode_weeks": args.episode_weeks,
                },
            )
            print(f"[✓] WandB run: {wb_run.url}")
        except ImportError:
            print("[!] wandb not installed — run: uv pip install wandb")
        except Exception as e:
            print(f"[!] WandB init failed: {e}")

    # Run episodes
=======
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
>>>>>>> origin/main
    from client import StaffingAgencyEnv
    from training.rollout import rollout_full_episode

    rng = random.Random(args.seed)
<<<<<<< HEAD
    all_profits: list[float] = []
    all_rewards: list[float] = []
    global_step = 0

    print(f"\n--- Inference: {args.num_episodes} episodes ---\n")

    for ep in range(args.num_episodes):
        ep_seed = rng.getrandbits(32)
        prompts = completions = step_rewards = None
        final_profit = cumulative_reward = 0.0

        for attempt in range(10):
            try:
                with StaffingAgencyEnv(base_url=args.env_url) as env:
                    prompts, completions, step_rewards, final_profit, cumulative_reward = \
                        rollout_full_episode(
                            env, model, tokenizer,
                            seed=ep_seed,
                            max_turns_per_step=args.max_turns_per_step,
                            rng=rng,
                            cfg=cfg,
                        )
                break
            except Exception as exc:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [!] Episode {ep} attempt {attempt+1} failed: {str(exc)[:80]}. Retry in {wait}s...")
                time.sleep(wait)

        if prompts is None:
            print(f"  [!] Episode {ep} failed after 10 attempts — skipping.")
            continue

        n_steps = len([p for p in prompts if p])
        all_profits.append(final_profit)
        all_rewards.append(cumulative_reward)

        print(f"  Episode {ep+1}/{args.num_episodes}  profit=${final_profit:>10,.0f}  "
              f"reward={cumulative_reward:>+10,.2f}  steps={n_steps}")

        # Per-step WandB: log every metric at every step so all charts have data
        if wb_run is not None and step_rewards:
            for i, r in enumerate(step_rewards):
                step_idx = global_step + i
                cum = sum(step_rewards[: i + 1])
                wb_run.log(
                    {
                        "infer/step_reward": r,
                        "infer/step_index": step_idx,
                        "infer/step_cumulative": cum,
                        "infer/episode": ep,
                        # Episode-level: include at every step so all charts have data
                        "infer/profit": final_profit,
                        "infer/cumulative_reward": cumulative_reward,
                        "infer/rollout_steps": n_steps,
                        "infer/num_episodes": ep + 1,
                    },
                    step=step_idx,
                )
            global_step += len(step_rewards)

    # Summary
    if all_profits:
        avg_profit = sum(all_profits) / len(all_profits)
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"\n--- Done ---")
        print(f"  Episodes: {len(all_profits)}")
        print(f"  Avg profit:  ${avg_profit:,.0f}")
        print(f"  Avg reward:  {avg_reward:+,.2f}")

        if wb_run is not None:
            wb_run.log(
                {
                    "infer/avg_profit": avg_profit,
                    "infer/avg_reward": avg_reward,
                    "infer/num_episodes": len(all_profits),
                },
                step=global_step,
            )
            wb_run.finish()
=======

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
>>>>>>> origin/main


if __name__ == "__main__":
    main()
