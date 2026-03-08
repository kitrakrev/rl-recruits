"""
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
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: uv pip install transformers torch peft")
        sys.exit(1)

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

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
    wb_run = None
    if args.wandb:
        try:
            import wandb
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
    from client import StaffingAgencyEnv
    from training.rollout import rollout_full_episode

    rng = random.Random(args.seed)
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


if __name__ == "__main__":
    main()
