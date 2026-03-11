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

import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Reduce CUDA memory fragmentation — critical for 8B models on 80GB GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if TYPE_CHECKING:
    from training.config import TrainingConfig


def train_grpo(cfg: "TrainingConfig") -> None:
    """
    Run the REINFORCE-GRPO training loop.

    Parameters
    ----------
    cfg : TrainingConfig — all hyperparameters. Build with TrainingConfig.from_args(args).
    """
    import os
    # Reduce CUDA memory fragmentation — PyTorch's own recommendation for OOM.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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

<<<<<<< HEAD
    # Enable gradient checkpointing to trade compute for VRAM.
    # Essential for 8B models on 80GB GPUs — saves ~50% activation memory.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
=======
    # Gradient checkpointing: recompute activations during backward instead of
    # storing them during forward — trades ~30% compute for large VRAM savings.
    model.enable_input_require_grads()   # needed by PEFT for gradient checkpointing
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
>>>>>>> origin/main

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
        # eval() + no_grad during rollout: avoids storing activations for 100+
        # generate() calls which would fill VRAM before training even starts.
        model.eval()
        torch.cuda.empty_cache()

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
        #
        # Gradient Accumulation design:
        #   micro_batch  = train_batch_size        (steps processed per backward call)
        #   accum_steps  = gradient_accumulation_steps  (micro-batches before optimizer.step)
        #   effective_bs = micro_batch × accum_steps   (true batch for one weight update)
        #
        # Memory strategy:
        #   - backward() called per STEP (not per batch) → only one graph alive at a time
        #   - loss scaled by 1/(micro_batch × accum_steps) for correct gradient magnitude
        #   - use_cache=False in forward → no KV-cache allocated during training
        #   - explicit del of all tensors + empty_cache() between optimizer steps
        accum_steps = cfg.gradient_accumulation_steps
        micro_bs    = cfg.train_batch_size
        eff_bs      = micro_bs * accum_steps

        print(f"\n  [Phase3] Policy update — {n_steps} steps  "
              f"micro_bs={micro_bs}  accum={accum_steps}  eff_bs={eff_bs}")
        torch.cuda.empty_cache()
        model.train()
        indices = list(range(n_steps))
        random.shuffle(indices)

        ep_loss_sum = ep_kl_sum = 0.0
        n_opt_steps = 0   # optimizer.step() calls (one per accum_steps micro-batches)
        skipped = 0

<<<<<<< HEAD
        for batch_start in range(0, len(indices), cfg.train_batch_size):
            batch_idx  = indices[batch_start : batch_start + cfg.train_batch_size]
            batch_loss_accum = 0.0  # scalar accumulator (no graph retention)
=======
        # Track accumulated gradients across micro-batches
        accum_loss   = 0.0   # scalar sum for logging
        accum_terms  = 0     # valid steps accumulated so far
        micro_count  = 0     # micro-batches since last optimizer.step
>>>>>>> origin/main

        optimizer.zero_grad()

        for batch_start in range(0, len(indices), micro_bs):
            micro_idx   = indices[batch_start : batch_start + micro_bs]
            micro_terms = 0
            micro_loss  = 0.0

            for idx in micro_idx:
                prompt_str     = prompts_v[idx]
                completion_str = completions_v[idx]
                adv            = float(advantages[idx].item())

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
                    del prompt_ids, full_ids
                    continue

                completion_ids = full_ids[:, comp_start:]

<<<<<<< HEAD
                # Reference log-probs FIRST (no grad) — compute before the
                # policy forward pass so the policy graph is the only thing
                # in memory when we call .backward().
                with torch.no_grad(), model.disable_adapter():
                    ref_logits = model(full_ids).logits[:, comp_start - 1 : -1, :]
                    ref_log_p  = F.log_softmax(ref_logits, dim=-1)
                    ref_tok_lp = ref_log_p.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)

                # Policy log-probs (LoRA adapters active)
                logits    = model(full_ids).logits[:, comp_start - 1 : -1, :]
=======
                # Policy log-probs (LoRA active, use_cache=False saves VRAM)
                logits    = model(full_ids, use_cache=False).logits[:, comp_start - 1 : -1, :]
>>>>>>> origin/main
                log_probs = F.log_softmax(logits, dim=-1)
                tok_lp    = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
                avg_log_p = tok_lp.mean()

<<<<<<< HEAD
                # Per-token KL: p * (log p − log q)
                kl = (tok_lp.exp() * (tok_lp - ref_tok_lp)).mean()
                ep_kl_sum += kl.item()

                step_loss  = (-adv * avg_log_p + cfg.kl_coeff * kl) / len(batch_idx)

                # Backward PER SAMPLE — gradients accumulate in .grad buffers
                # but the computation graph is freed immediately, so only ONE
                # sample's activations live in VRAM at a time.
                step_loss.backward()
                batch_loss_accum += step_loss.item()

                # Free intermediate tensors before next sample
                del logits, log_probs, tok_lp, ref_logits, ref_log_p, ref_tok_lp
                del full_ids, prompt_ids, completion_ids
                torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            ep_loss_sum += batch_loss_accum
            all_losses.append(batch_loss_accum)
            n_batches += 1
=======
                # Reference log-probs: base model (LoRA disabled, no graph)
                with torch.no_grad(), model.disable_adapter():
                    ref_logits = model(full_ids, use_cache=False).logits[:, comp_start - 1 : -1, :]
                    ref_log_p  = F.log_softmax(ref_logits, dim=-1)
                    ref_tok_lp = ref_log_p.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1).detach()

                kl = (tok_lp.exp() * (tok_lp - ref_tok_lp)).mean()
                ep_kl_sum += kl.item()

                # Scale by effective batch size so gradient magnitude is independent
                # of both micro_bs and accum_steps
                step_loss = (-adv * avg_log_p + cfg.kl_coeff * kl) / eff_bs

                # Backward immediately — only ONE graph alive at a time
                step_loss.backward()
                micro_loss  += step_loss.item()
                micro_terms += 1

                del logits, log_probs, tok_lp, avg_log_p
                del ref_logits, ref_log_p, ref_tok_lp, kl, step_loss
                del prompt_ids, full_ids, completion_ids
>>>>>>> origin/main

            accum_loss  += micro_loss
            accum_terms += micro_terms
            micro_count += 1

            # ---- Optimizer step every accum_steps micro-batches ----
            is_last_micro = (batch_start + micro_bs) >= len(indices)
            should_step   = (micro_count % accum_steps == 0) or is_last_micro

            if should_step and accum_terms > 0:
                total_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                ep_loss_sum += accum_loss
                all_losses.append(accum_loss)
                n_opt_steps += 1

                print(f"  [Phase3] opt_step {n_opt_steps:3d}  "
                      f"micro_batches={micro_count}  terms={accum_terms}  "
                      f"loss={accum_loss:+.6f}  grad_norm={total_norm:.4f}")

                if wb_run is not None:
                    wb_run.log({
                        "train/loss":      accum_loss,
                        "train/grad_norm": total_norm,
                        "train/kl":        ep_kl_sum / max(1, accum_terms),
                        "train/opt_step":  n_opt_steps,
                        "episode":         ep,
                    })

                # Reset accumulators and free VRAM between optimizer steps
                accum_loss  = 0.0
                accum_terms = 0
                micro_count = 0
                torch.cuda.empty_cache()

            elif should_step and accum_terms == 0:
                # All micro-batches in this accumulation window were skipped
                optimizer.zero_grad()
                micro_count = 0

        avg_ep_loss = ep_loss_sum / max(1, n_opt_steps)
        avg_ep_kl   = ep_kl_sum   / max(1, n_steps)

        # ---- Record & log ------------------------------------------------
        all_profits.append(final_profit)
        all_env_rewards.append(cumulative_env_reward)

        print(
            f"\n  === Episode {ep:4d}/{cfg.num_episodes}  "
            f"profit=${final_profit:>10,.0f}  "
            f"env_reward={cumulative_env_reward:+10,.2f}  "
            f"steps={n_steps}  skipped={skipped}  opt_steps={n_opt_steps}  "
            f"loss={avg_ep_loss:.6f}  kl={avg_ep_kl:.6f} ==="
        )

        if wb_run is not None:
            step_rewards_list = [float(r) for r in step_rewards_v]
            wb_run.log({
                "episode":         ep,
                "profit":          final_profit,
                "env_reward":      cumulative_env_reward,
                "loss":            avg_ep_loss,
                "kl":              avg_ep_kl,
                "rollout_steps":   n_steps,
                "skipped_steps":   skipped,
                "n_opt_steps":     n_opt_steps,
                "mean_advantage":  advantages.mean().item(),
                "std_advantage":   advantages.std().item(),
                "return_mean":     returns_t.mean().item(),
                "return_min":      returns_t.min().item(),
                "return_max":      returns_t.max().item(),
                # Per-step reward stats (so WandB charts show reward distribution)
                "step_reward_mean": sum(step_rewards_list) / max(1, len(step_rewards_list)),
                "step_reward_min":  min(step_rewards_list) if step_rewards_list else 0.0,
                "step_reward_max":  max(step_rewards_list) if step_rewards_list else 0.0,
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
