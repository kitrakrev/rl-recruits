"""
Training hyperparameters for the REINFORCE-GRPO loop.

All environment constants live in env/config.py → Config.
This module is intentionally free of any env-side imports.

Configurable via:
  - Constructor:  TrainingConfig(learning_rate=1e-5)
  - YAML file:    TrainingConfig.from_yaml("training/config.yaml")
  - CLI args:     TrainingConfig.from_args(parsed_args)
  - API:          GET /config/training  (read-only view)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any


@dataclass
class TrainingConfig:
    # --- Model ---
    # Qwen3-0.6B: lightweight for fast iteration / smoke testing.
    # Same architecture as Qwen3-8B — enable_thinking=False, <tool_call> format,
    # and LoRA all apply identically. Upgrade to Qwen/Qwen3-8B for production.
    model_name: str = "Qwen/Qwen3-0.6B"

    # --- LoRA ---
    lora_rank: int = 16             # adapter rank (higher = more capacity)
    lora_alpha: int = 32            # scaling factor = lora_alpha / lora_rank
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # --- Optimisation ---
    learning_rate: float = 5e-6
    gamma: float = 0.99             # discount factor for returns
    kl_coeff: float = 0.05          # KL penalty weight (prevents policy drift)
    train_batch_size: int = 4       # steps per gradient accumulation batch
    max_grad_norm: float = 1.0      # gradient clipping threshold

    # --- Generation ---
    max_turns_per_step: int = 10    # max tool calls per week before auto-advance
    max_prompt_len: int = 4096      # left-truncation limit for prompt tokens
                                    # (system prompt + 15 tools ≈ 2500 tokens alone;
                                    #  2048 was too small for 8B and truncated
                                    #  the system prompt, causing garbage output)
    max_full_len: int = 4608        # truncation limit for prompt + completion
    max_new_tokens: int = 512       # generation budget per turn
    # Qwen3 non-thinking mode recommended settings (HF model card)
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    # --- Experiment ---
    num_episodes: int = 200
    seed: int = 42
    output_dir: str = "training/checkpoints"
    env_url: str = "http://localhost:8000"

    # --- W&B ---
    wandb_project: str = "kanandan-university-of-california/Staffing_agent"
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))

    # --- Dry-run ---
    dry_run: bool = False

    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load from a YAML file; only keys matching dataclass fields are used."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: uv pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_args(cls, args: Any) -> "TrainingConfig":
        """Build from an argparse Namespace, overriding only non-None fields."""
        cfg = cls()
        for f in dc_fields(cfg):
            val = getattr(args, f.name, None)
            if val is not None:
                setattr(cfg, f.name, val)
        return cfg

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}
