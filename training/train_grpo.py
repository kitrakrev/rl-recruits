"""
Staffing Agency — REINFORCE-GRPO Training Entry Point
======================================================

Run (full training, GPU required):
    uv run python training/train_grpo.py \\
        --env_url http://localhost:8000 \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
        --num_episodes 200 \\
        --output_dir training/checkpoints \\
        --wandb_api_key YOUR_KEY

Run (dry run — no GPU, validates full HTTP stack):
    uv run python training/train_grpo.py --dry_run --num_episodes 90

Load hyperparameters from YAML (overridable by CLI flags):
    uv run python training/train_grpo.py --config training/config.yaml

All hyperparameters live in env/config.py → TrainingConfig.
All constants (penalties, reward scaling) live in env/config.py → Config.
The environment config is hot-patchable at runtime via PATCH /config/env.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / dry-run the Staffing Agency RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument("--env_url",    default=None, help="Environment server URL")
    p.add_argument("--model_name", default=None, help="HuggingFace model ID")
    p.add_argument("--num_episodes", type=int, default=None, help="Training episodes")
    p.add_argument("--output_dir", default=None, help="Where to save the fine-tuned model")
    p.add_argument("--seed",       type=int, default=None)

    # Dry-run
    p.add_argument("--dry_run", action="store_true",
                   help="Run heuristic policies without a model (validates HTTP stack)")
    p.add_argument("--max_turns", type=int, default=None,
                   help="Tool calls per episode step (dry-run) or per week (training)")

    # Hyperparameters (all optional — defaults come from TrainingConfig)
    p.add_argument("--learning_rate",     type=float, default=None)
    p.add_argument("--gamma",             type=float, default=None, help="Discount factor")
    p.add_argument("--kl_coeff",          type=float, default=None, help="KL penalty weight")
    p.add_argument("--train_batch_size",  type=int,   default=None)
    p.add_argument("--temperature",       type=float, default=None)
    p.add_argument("--max_turns_per_step",type=int,   default=None)

    # W&B
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_api_key", default=os.getenv("WANDB_API_KEY"),
                   help="W&B API key (falls back to WANDB_API_KEY env var)")

    # Config file
    p.add_argument("--config", default=None, metavar="YAML",
                   help="Path to a YAML file with TrainingConfig values "
                        "(CLI flags override YAML values)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path("training").mkdir(exist_ok=True)

    from env.config import TrainingConfig

    # Build config: YAML base (if given) then CLI overrides
    if args.config:
        cfg = TrainingConfig.from_yaml(args.config)
        # Apply CLI flags on top of YAML
        cfg = TrainingConfig.from_args(args)  # from_args merges non-None values
    else:
        cfg = TrainingConfig.from_args(args)

    # --dry_run takes precedence
    if args.dry_run or cfg.dry_run:
        from training.dry_run import dry_run_simulate
        dry_run_simulate(
            env_url=cfg.env_url,
            num_episodes=cfg.num_episodes,
            max_turns=args.max_turns or cfg.max_turns_per_step,
            seed=cfg.seed,
        )
    else:
        from training.reinforce import train_grpo
        train_grpo(cfg)


if __name__ == "__main__":
    main()
