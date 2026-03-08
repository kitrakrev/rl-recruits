"""
Episode rollout functions.

rollout_full_episode  — live 52-week rollout using model.generate().
                        Returns per-step rewards for the REINFORCE update.

rollout_episode       — OpenEnv-compatible rollout that accepts any callable
                        generate function (useful for testing / integration).
"""
from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Callable

from training.prompts import SYSTEM_PROMPT, TOOLS, parse_tool_call

if TYPE_CHECKING:
    from env.config import TrainingConfig


# ---------------------------------------------------------------------------
# Full 52-week rollout (used by the REINFORCE training loop)
# ---------------------------------------------------------------------------

def rollout_full_episode(
    env,
    model,
    tokenizer,
    seed: int,
    max_turns_per_step: int = 10,
    rng: random.Random | None = None,
    cfg: "TrainingConfig | None" = None,
) -> tuple[list[str], list[str], list[float], float, float]:
    """
    Run one complete 52-week episode with the current model.

    At each turn the model generates a tool call, which is executed against the
    live environment server. The env returns a per-step reward that encodes
    real business consequences (interview costs, onboarding, billing margins,
    expiry penalties, client churn, etc.). All rewards come from env.step() —
    no fabricated rewards.

    Parameters
    ----------
    env         : StaffingAgencyEnv context manager (already opened)
    model       : HuggingFace AutoModelForCausalLM
    tokenizer   : matching AutoTokenizer
    seed        : episode seed (controls env stochasticity)
    max_turns_per_step : max tool calls per week before auto-advancing
    rng         : optional Random instance for tool-call IDs; creates one if None
    cfg         : TrainingConfig for generation / tokenisation settings

    Returns
    -------
    (prompts, completions, step_rewards, final_profit, cumulative_env_reward)
    """
    import torch

    from models import StaffingAction  # project root

    if rng is None:
        rng = random.Random(seed)

    # Resolve generation settings from cfg or use safe defaults
    max_prompt_len = cfg.max_prompt_len if cfg else 2048
    max_full_len   = cfg.max_full_len   if cfg else 2560
    max_new_tokens = cfg.max_new_tokens if cfg else 512
    temperature    = cfg.temperature    if cfg else 0.8

    env.reset(seed=seed)
    prompts_out: list[str]   = []
    completions_out: list[str] = []
    step_rewards: list[float]  = []
    cumulative_env_reward: float = 0.0
    final_profit = 0.0

    # Seed Week 1: give the model real IDs so it can act immediately.
    init_state: dict = {}
    try:
        r = env.step(StaffingAction(tool="get_financial_summary", params={}))
        init_state["financials"] = r.observation.tool_result or {}
        final_profit = float(r.observation.profit or 0.0)
    except Exception:
        pass
    try:
        r = env.step(StaffingAction(tool="get_candidate_state", params={}))
        init_state["candidates"] = r.observation.tool_result or {}
    except Exception:
        pass
    try:
        r = env.step(StaffingAction(tool="get_client_state", params={}))
        init_state["clients"] = r.observation.tool_result or {}
    except Exception:
        pass

    init_str = json.dumps(init_state, indent=2)[:1200]
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Week 1 of 52. Current state:\n{init_str}\n\n"
            "Take business actions. Find candidates, interview them, hire the best ones, "
            "confirm projects, and match candidates to project roles to generate profit. "
            "Call advance_week when done."
        )},
    ]

    print(f"\n      --- Rollout Phase (Seed {seed}) ---")

    prev_profit = final_profit
    for week in range(1, 53):
        week_advanced = False
        for turn in range(1, max_turns_per_step + 1):
            prompt_str = tokenizer.apply_chat_template(
                conversation,
                tools=TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )

            if week == 1 and turn == 1:
                print(f"DEBUG: First rendered prompt (truncated):\n{prompt_str[:800]}...")

            input_ids = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )

            completion_ids = out_ids[0][input_ids.input_ids.shape[-1]:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Decode full output so we catch tool calls pre-filled by the chat template
            full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            parsed = parse_tool_call(full_text) or parse_tool_call(completion_text)

            reward = 0.0
            final_cash = 0.0
            tool_result_text = "No tool parsed — no action taken."

            if parsed:
                tool_name, params = parsed
                try:
                    result = env.step(StaffingAction(tool=tool_name, params=params))
                    reward = float(result.reward or 0.0)
                    final_profit = float(result.observation.profit or 0.0)
                    final_cash = float(result.observation.cash or 0.0)
                    tr = result.observation.tool_result or {}
                    tool_result_text = json.dumps(tr, indent=2)[:400]
                    profit_delta = final_profit - prev_profit
                    prev_profit = final_profit

                    status = "PROFIT" if profit_delta > 0 else ("LOSS" if profit_delta < 0 else "NEUTRAL")
                    print(
                        f"      Week {week:2d} Turn {turn:2d}: {tool_name:25s} | "
                        f"{status:7s} | Cash: ${final_cash:>9,.0f} | "
                        f"Profit: ${final_profit:>9,.0f} | ΔProfit: ${profit_delta:+,.0f} | "
                        f"Reward: {reward:+.2f}"
                    )

                    if tool_name == "advance_week":
                        week_advanced = True
                except Exception as e:
                    tool_result_text = f"Error: {e}"
            else:
                clean = completion_text.strip().replace("\n", " ")
                print(
                    f"      Week {week:2d} Turn {turn:2d}: [invalid syntax]            | "
                    f"FAILED  | Cash: ${final_cash:>9,.0f} | "
                    f"Profit: ${final_profit:>9,.0f} | Reward: +0.00"
                )
                print(f"        DEBUG RAW: {clean[:120]}...")

            prompts_out.append(prompt_str)
            completions_out.append(completion_text)
            step_rewards.append(reward)
            cumulative_env_reward += reward

            if week_advanced:
                break

            # Build next conversation turn
            tool_call_id = f"call_{rng.getrandbits(32):08x}"
            assistant_msg: dict = {"role": "assistant", "content": completion_text}
            if parsed:
                assistant_msg["tool_calls"] = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": parsed[0], "arguments": parsed[1]},
                }]
            conversation.append(assistant_msg)

            if parsed:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": parsed[0],
                    "content": tool_result_text,
                })
            else:
                conversation.append({
                    "role": "user",
                    "content": (
                        "No tool call found. Call one of the available tools using "
                        'the native format, e.g.:\n<tool_call>\n{"name": "get_agency_state", '
                        '"arguments": {}}\n</tool_call>'
                    ),
                })

        # Auto-advance if model never called advance_week
        if not week_advanced:
            print(f"      Week {week:2d}: [Max turns reached — auto-advancing]")
            try:
                result = env.step(StaffingAction(tool="advance_week", params={}))
                final_profit = float(result.observation.profit or 0.0)
                # Capture the auto-advance reward too
                auto_reward = float(result.reward or 0.0)
                step_rewards.append(auto_reward)
                cumulative_env_reward += auto_reward
                prompts_out.append("")       # placeholder — no model output for auto-advance
                completions_out.append("")
            except Exception:
                pass

        # Trim history and inject fresh state for next week
        system_msg = conversation[0]
        recent = conversation[-6:] if len(conversation) > 7 else conversation[1:]

        week_state: dict = {}
        try:
            r = env.step(StaffingAction(tool="get_candidate_state", params={}))
            week_state["candidates"] = r.observation.tool_result or {}
        except Exception:
            pass
        try:
            r = env.step(StaffingAction(tool="get_client_state", params={}))
            week_state["clients"] = r.observation.tool_result or {}
        except Exception:
            pass

        state_str = json.dumps(week_state, indent=2)[:1000]
        conversation = [system_msg] + recent + [{
            "role": "user",
            "content": (
                f"[Week {week + 1} of 52 — current state]\n{state_str}\n\n"
                "Continue: hire candidates, fill project roles, then call advance_week."
            ),
        }]

    print(f"      --- Episode Outcome: Profit=${final_profit:,.0f}  EnvReward={cumulative_env_reward:+,.2f} ---")
    return prompts_out, completions_out, step_rewards, final_profit, cumulative_env_reward


# ---------------------------------------------------------------------------
# OpenEnv-compatible rollout (generic generate_fn interface)
# ---------------------------------------------------------------------------

def rollout_episode(
    env_client,
    model_generate_fn: Callable,
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = 200,
    seed: int | None = None,
) -> tuple[list[dict], float]:
    """
    Generic episode rollout compatible with the OpenEnv CallToolAction API.

    Parameters
    ----------
    env_client       : OpenEnv client (supports .reset() and .step())
    model_generate_fn : callable(prompt_ids) → {"completion_ids": tensor, "logprobs": ...}
    tokenizer        : matching tokenizer
    system_prompt    : system message text
    max_turns        : maximum tool calls per episode
    seed             : optional seed passed to env.reset()

    Returns
    -------
    (rollout_data, cumulative_reward)
    rollout_data is a list of dicts with keys:
        prompt_ids, completion_ids, logprobs, reward, tool_name
    """
    try:
        from openenv.core.env_server.mcp_types import CallToolAction
    except ImportError:
        raise ImportError("openenv-core required for rollout_episode")

    reset_result = env_client.reset(seed=seed)
    obs_text = (
        reset_result.observation.metadata.get("message", "")
        if hasattr(reset_result.observation, "metadata")
        else str(reset_result)
    )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Episode started. Current state:\n{obs_text}\n\nWhat is your first action?"},
    ]

    rollout_data: list[dict] = []
    cumulative_reward = 0.0

    for turn in range(max_turns):
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")

        outputs = model_generate_fn(prompt_ids)
        completion_ids = outputs["completion_ids"]
        logprobs = outputs.get("logprobs")
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        parsed = parse_tool_call(completion_text)
        reward = 0.0
        done = False

        if parsed:
            tool_name, params = parsed
            try:
                action = CallToolAction(tool_name=tool_name, arguments=params)
                result = env_client.step(action)
                reward = float(result.reward or 0.0)
                cumulative_reward += reward
                obs_meta = (
                    result.observation.metadata
                    if hasattr(result.observation, "metadata")
                    else {}
                )
                tool_output = json.dumps(obs_meta.get("tool_result", obs_meta), indent=2)
                done = result.done
            except Exception as e:
                tool_output = f"Error: {e}"
        else:
            tool_output = (
                "No tool call found. Use the native format:\n"
                '<tool_call>\n{"name": "tool_name", "arguments": {}}\n</tool_call>'
            )

        conversation.append({"role": "assistant", "content": completion_text})
        conversation.append({
            "role": "user",
            "content": (
                f"Tool result:\n{tool_output}\n"
                f"Step reward: {reward:.4f} | Cumulative: {cumulative_reward:.4f}\n"
                + ("Episode DONE." if done else "Continue. What is your next action?")
            ),
        })

        rollout_data.append({
            "prompt_ids": prompt_ids[0].tolist(),
            "completion_ids": completion_ids[0].tolist(),
            "logprobs": logprobs,
            "reward": reward,
            "tool_name": parsed[0] if parsed else "parse_error",
        })

        if done:
            break

    return rollout_data, cumulative_reward
