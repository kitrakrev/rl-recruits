"""
REINFORCE-style GRPO training loop with online environment rewards.

Each training iteration:
  1. ROLLOUT  — Play one full 52-week episode against the live environment.
     At each step the model generates a tool call; env.step() executes it and
     returns a per-step reward encoding real business consequences (interview
     costs, bench burn, billing margins, expiry penalties, client churn, etc.).
     ALL rewards come from env interaction — no fabricated rewards.

  2. RETURNS  — Compute discounted cumulative returns from per-step rewards.
     return_t = reward_t + gamma * return_{t+1}
     Advantages are z-score normalised across the trajectory.

  3. UPDATE   — REINFORCE policy gradient with KL penalty:
     loss = −advantage * log_prob(completion | prompt) + kl_coeff * KL
     The KL penalty (against a frozen reference model) prevents the policy
     from diverging too far from the pre-trained checkpoint.

  4. REPEAT   — Loop. As the model improves, profit and env reward rise.

Why not TRL's GRPOTrainer?
  TRL re-generates completions and re-evaluates them with the reward function.
  For a stateful env, the state from the original rollout is gone, so every
  re-generated completion would receive the same cached reward → advantage = 0
  → zero gradient → no learning. This custom loop couples reward evaluation
  with the rollout phase when the env IS in the correct state.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.config import TrainingConfig


def train_grpo(cfg: "TrainingConfig") -> None:
    """
    Run the REINFORCE-GRPO training loop.

    Parameters
    ----------
    cfg : TrainingConfig — all hyperparameters. Build with TrainingConfig.from_args(args).
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: uv pip install transformers torch peft")
        sys.exit(1)

    try:
        import requests as _req
        _req.get(f"{cfg.env_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"[✗] Environment server not reachable at {cfg.env_url}: {e}")
        print("    Start it with: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Lazy import here so the module is importable without optional dependencies
    from client import StaffingAgencyEnv
    from training.rollout import rollout_full_episode
    from training.metrics import plot_reward_curves, save_metrics

    # ------------------------------------------------------------------
    # W&B setup
    # ------------------------------------------------------------------
    wb_run = None
    try:
        import wandb
        if cfg.wandb_api_key:
            wandb.login(key=cfg.wandb_api_key, relogin=True)
            entity, project = (cfg.wandb_project.split("/", 1) + ["Staffing_agent"])[:2]
            wb_run = wandb.init(
                entity=entity,
                project=project,
                config=cfg.to_dict(),
            )
            print(f"[✓] W&B run: {wb_run.url}")
        else:
            print("[!] No WANDB_API_KEY — skipping W&B logging")
    except ImportError:
        print("[!] wandb not installed — skipping W&B logging")

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Wrap with LoRA — only adapter weights are trained (~0.5% of params).
    # This keeps total VRAM well under 80GB even on 8B models, and eliminates
    # the need for a separate frozen reference model: we compute KL by
    # temporarily disabling the adapters on the SAME model object.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Only optimise the LoRA adapter parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
    )

    print(f"\nStarting REINFORCE-GRPO training")
    print(f"  model:    {cfg.model_name}  (LoRA r={cfg.lora_rank} α={cfg.lora_alpha})")
    print(f"  env:      {cfg.env_url}")
    print(f"  episodes: {cfg.num_episodes}  (each = 52-week rollout + policy update)")
    print(f"  gamma:    {cfg.gamma}  kl_coeff: {cfg.kl_coeff}  lr: {cfg.learning_rate}\n")

    rng = random.Random(cfg.seed)
    all_profits:     list[float] = []
    all_env_rewards: list[float] = []
    all_losses:      list[float] = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for ep in range(cfg.num_episodes):
        ep_seed = rng.getrandbits(32)

        # ---- Phase 1: Rollout against live env ---------------------------
        prompts = completions = step_rewards = None
        final_profit = cumulative_env_reward = 0.0

        for attempt in range(10):
            try:
                with StaffingAgencyEnv(base_url=cfg.env_url) as env:
                    prompts, completions, step_rewards, final_profit, cumulative_env_reward = \
                        rollout_full_episode(
                            env, model, tokenizer,
                            seed=ep_seed,
                            max_turns_per_step=cfg.max_turns_per_step,
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

        # Filter out auto-advance placeholders (empty prompt/completion)
        valid = [(p, c, r) for p, c, r in zip(prompts, completions, step_rewards) if p]
        print(f"\n  [Phase2] total steps={len(step_rewards)}  valid={len(valid)}  "
              f"raw_rewards min={min(step_rewards):.2f} max={max(step_rewards):.2f} "
              f"sum={sum(step_rewards):.2f}")
        if not valid:
            print(f"  [!] Episode {ep} produced 0 valid steps — skipping.")
            continue

        prompts_v, completions_v, step_rewards_v = zip(*valid)
        n_steps = len(prompts_v)

        # ---- Phase 2: Discounted returns + advantage normalisation -------
        returns: list[float] = [0.0] * n_steps
        G = 0.0
        for t in range(n_steps - 1, -1, -1):
            G = step_rewards_v[t] + cfg.gamma * G
            returns[t] = G

        returns_t   = torch.tensor(returns, dtype=torch.float32)
        advantages  = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        print(f"  [Phase2] returns  min={returns_t.min():.2f} max={returns_t.max():.2f} mean={returns_t.mean():.2f}")
        print(f"  [Phase2] advantages min={advantages.min():.4f} max={advantages.max():.4f} "
              f"std={advantages.std():.4f}  nonzero={int((advantages.abs() > 1e-6).sum())}/{n_steps}")

        # ---- Phase 3: REINFORCE policy gradient update -------------------
        print(f"\n  [Phase3] Starting policy update — {n_steps} steps, "
              f"batch_size={cfg.train_batch_size}")
        model.train()
        indices = list(range(n_steps))
        random.shuffle(indices)

        ep_loss_sum = ep_kl_sum = 0.0
        n_batches = 0
        skipped = 0

        for batch_start in range(0, len(indices), cfg.train_batch_size):
            batch_idx  = indices[batch_start : batch_start + cfg.train_batch_size]
            batch_loss = torch.tensor(0.0, requires_grad=False, device=model.device)
            batch_terms = 0

            for idx in batch_idx:
                prompt_str     = prompts_v[idx]
                completion_str = completions_v[idx]
                adv            = advantages[idx].item()

                if not completion_str.strip():
                    skipped += 1
                    continue

                prompt_ids = tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_prompt_len,
                ).input_ids.to(model.device)

                full_ids = tokenizer(
                    prompt_str + completion_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_full_len,
                ).input_ids.to(model.device)

                comp_start = prompt_ids.shape[1]
                if full_ids.shape[1] <= comp_start:
                    skipped += 1
                    continue  # completion was entirely truncated

                completion_ids = full_ids[:, comp_start:]

                # Policy log-probs (LoRA adapters active)
                logits    = model(full_ids).logits[:, comp_start - 1 : -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                tok_lp    = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
                avg_log_p = tok_lp.mean()

                # Reference log-probs: base model = disable LoRA adapters temporarily.
                with torch.no_grad(), model.disable_adapter():
                    ref_logits = model(full_ids).logits[:, comp_start - 1 : -1, :]
                    ref_log_p  = F.log_softmax(ref_logits, dim=-1)
                    ref_tok_lp = ref_log_p.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)

                # Per-token KL: p * (log p − log q)
                kl = (tok_lp.exp() * (tok_lp - ref_tok_lp)).mean()
                ep_kl_sum += kl.item()

                step_loss  = -adv * avg_log_p + cfg.kl_coeff * kl
                batch_loss = batch_loss + step_loss / len(batch_idx)
                batch_terms += 1

            if batch_terms == 0:
                print(f"  [Phase3] batch {n_batches} skipped (all truncated)")
                continue

            pre_grad_check = batch_loss.requires_grad
            batch_loss.backward()

            # Gradient norm across all trainable params
            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = batch_loss.item()
            ep_loss_sum += loss_val
            all_losses.append(loss_val)
            n_batches += 1

            print(f"  [Phase3] batch {n_batches:3d}  terms={batch_terms}  "
                  f"loss={loss_val:+.6f}  grad_norm={total_norm:.4f}  "
                  f"requires_grad={pre_grad_check}")

            if wb_run is not None:
                wb_run.log({
                    "batch/loss":      loss_val,
                    "batch/grad_norm": total_norm,
                    "batch/kl":        ep_kl_sum / max(1, (batch_start + len(batch_idx))),
                    "episode":         ep,
                })

        avg_ep_loss = ep_loss_sum / max(1, n_batches)
        avg_ep_kl   = ep_kl_sum   / max(1, n_steps)

        # ---- Record & log ------------------------------------------------
        all_profits.append(final_profit)
        all_env_rewards.append(cumulative_env_reward)

        print(
            f"\n  === Episode {ep:4d}/{cfg.num_episodes}  "
            f"profit=${final_profit:>10,.0f}  "
            f"env_reward={cumulative_env_reward:+10,.2f}  "
            f"steps={n_steps}  skipped={skipped}  batches={n_batches}  "
            f"loss={avg_ep_loss:.6f}  kl={avg_ep_kl:.6f} ==="
        )

        if wb_run is not None:
            wb_run.log({
                "episode":         ep,
                "profit":          final_profit,
                "env_reward":      cumulative_env_reward,
                "loss":            avg_ep_loss,
                "kl":              avg_ep_kl,
                "rollout_steps":   n_steps,
                "skipped_steps":   skipped,
                "n_batches":       n_batches,
                "mean_advantage":  advantages.mean().item(),
                "std_advantage":   advantages.std().item(),
                "return_mean":     returns_t.mean().item(),
                "return_min":      returns_t.min().item(),
                "return_max":      returns_t.max().item(),
            })

    # ------------------------------------------------------------------
    # Save model + plots
    # ------------------------------------------------------------------
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # save_pretrained on a PeftModel saves only the small LoRA adapter weights.
    # To load later: model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\n[✓] Training complete. LoRA adapter saved to {cfg.output_dir}")

    if wb_run is not None:
        wb_run.finish()

    results = {
        "grpo_mc_training": {
            "rewards": all_env_rewards,
            "profits": all_profits,
            "losses":  all_losses,
        }
    }
    plot_reward_curves(results, len(all_profits))
    save_metrics(results)
