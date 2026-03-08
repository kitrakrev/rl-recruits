"""
GRPO Training Script — Staffing Agency OpenEnv
================================================
Uses TRL's GRPOTrainer + OpenEnv rollout function to train an LLM to act
as a staffing agency CEO that maximises profit over 52-week episodes.

Theme alignment:
  - Multi-Agent: agent manages multiple clients + candidates simultaneously
  - Long-Horizon: 52-step sparse reward, multi-role project sealing
  - Scale AI / Mercor sub-theme: HR/business long-horizon workflow

Run:
    uv run python training/train_grpo.py \
        --env_url http://localhost:8000 \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --num_episodes 200 \
        --max_turns 10

Reward curves saved to: training/reward_curves.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_url", default="http://localhost:8000")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--max_turns", type=int, default=10,
                   help="Tool calls per episode step")
    p.add_argument("--output_dir", default="training/checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Run without TRL (simulate rewards for testing pipeline)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a staffing agency CEO managing a recruiting business.
Your goal is to maximise profit over 52 business weeks by:
1. Finding and interviewing candidates from the market
2. Hiring top candidates and placing them on client projects
3. Managing client relationships and filling projects before deadlines
4. Balancing bench costs (hired-but-unplaced = salary drain) vs revenue

CRITICAL ECONOMICS:
- Placed candidate = +$265 to +$625/week MARGIN (profit)
- Benched candidate = -$1,058 to -$2,500/week BURN
- Hiring costs $2,000 onboarding upfront
- Projects must be FULLY filled to generate any revenue
- Expired projects = large penalty
- Client churn (satisfaction < 0.3) = $50,000 LTV penalty

STRATEGY HINTS:
- Always check market demand before hiring (get_market_demand)
- Confirm projects before committing candidates (confirm_project)
- High-rating candidates earn more but cost more to bench — place them fast
- Pass on projects you cannot fill rather than letting them expire

Use the available tools to manage your agency. Think step by step.
When you want to call a tool, output EXACTLY:
TOOL: <tool_name>
PARAMS: <json_params_or_empty_dict>

Example:
TOOL: get_agency_state
PARAMS: {}

TOOL: find_candidate
PARAMS: {"developer_type": "backend"}

TOOL: match_candidate_to_project
PARAMS: {"candidate_id": "C-BA-abc12345", "project_id": "P-CL-001-xyz", "role_id": "R-P-CL-001-xyz-0"}
"""


# ---------------------------------------------------------------------------
# Rollout logic (env interaction)
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse TOOL: / PARAMS: from model output."""
    lines = text.strip().split("\n")
    tool_name = None
    params = {}
    for i, line in enumerate(lines):
        if line.startswith("TOOL:"):
            tool_name = line.replace("TOOL:", "").strip()
        elif line.startswith("PARAMS:"):
            params_str = line.replace("PARAMS:", "").strip()
            try:
                params = json.loads(params_str) if params_str and params_str != "{}" else {}
            except json.JSONDecodeError:
                params = {}
    return (tool_name, params) if tool_name else None


def rollout_episode(env_client, model_generate_fn, tokenizer, system_prompt, max_turns, seed=None):
    """
    Run one full episode rollout.
    Returns: list of (prompt_ids, completion_ids, logprobs, reward) per turn.
    """
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    reset_result = env_client.reset(seed=seed)
    obs_text = reset_result.observation.metadata.get("message", "") if hasattr(reset_result.observation, 'metadata') else str(reset_result)

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Episode started. Current state:\n{obs_text}\n\nWhat is your first action?"},
    ]

    rollout_data = []
    cumulative_reward = 0.0

    for turn in range(max_turns):
        # Tokenize prompt
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")

        # Generate completion
        outputs = model_generate_fn(prompt_ids)
        completion_ids = outputs["completion_ids"]
        logprobs = outputs.get("logprobs", None)
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        # Parse tool call from completion
        parsed = parse_tool_call(completion_text)
        reward = 0.0

        if parsed:
            tool_name, params = parsed
            try:
                action = CallToolAction(tool_name=tool_name, arguments=params)
                result = env_client.step(action)
                reward = float(result.reward or 0.0)
                cumulative_reward += reward
                obs_meta = result.observation.metadata if hasattr(result.observation, 'metadata') else {}
                tool_output = json.dumps(obs_meta.get("tool_result", obs_meta), indent=2)
                done = result.done
            except Exception as e:
                tool_output = f"Error: {e}"
                done = False
        else:
            tool_output = "Could not parse tool call. Use TOOL: / PARAMS: format."
            done = False

        # Add to conversation
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


# ---------------------------------------------------------------------------
# Reward functions for TRL GRPOTrainer
# ---------------------------------------------------------------------------

def reward_fn_profit(completions: list[str], **kwargs) -> list[float]:
    """Primary reward: cumulative episode profit signal."""
    rewards = kwargs.get("env_reward", [])
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


def reward_fn_tool_format(completions: list[str], **kwargs) -> list[float]:
    """Shaping reward: +0.1 for correctly formatted TOOL: / PARAMS: output."""
    rewards = []
    for c in completions:
        if "TOOL:" in c and "PARAMS:" in c:
            parsed = parse_tool_call(c)
            rewards.append(0.1 if parsed else -0.05)
        else:
            rewards.append(-0.1)  # penalise free-text without tool call
    return rewards


def reward_fn_placement(completions: list[str], **kwargs) -> list[float]:
    """Shaping reward: bonus for match_candidate_to_project calls."""
    rewards = []
    for c in completions:
        parsed = parse_tool_call(c)
        if parsed and parsed[0] == "match_candidate_to_project":
            rewards.append(0.2)
        elif parsed and parsed[0] in ("hire_candidate", "interview_candidate"):
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dry-run simulator (no TRL/GPU needed — for pipeline testing)
# ---------------------------------------------------------------------------

def dry_run_simulate(env_url: str, num_episodes: int, max_turns: int, seed: int):
    """
    Simulate training without a model. Uses heuristic policies to demonstrate
    reward curves and validate the FULL HTTP stack (server → MCP tools → env).

    Hits the live server at env_url — make sure it's running:
        uvicorn server.app:app --host 0.0.0.0 --port 8000
    """
    import random
    import requests as _req

    print("\n" + "="*60)
    print("DRY RUN MODE — Simulating reward curves via live server")
    print(f"Server: {env_url}")
    print("="*60)

    # Verify server is up
    try:
        _req.get(f"{env_url}/health", timeout=5).raise_for_status()
        print("[✓] Server is healthy")
    except Exception as e:
        print(f"[✗] Server not reachable at {env_url}: {e}")
        print("    Start it with: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    from models import StaffingAction
    from client import StaffingAgencyEnv

    rng = random.Random(seed)

    policies = {
        "random":            _policy_random_http,
        "greedy":            _policy_greedy_http,
        "optimal_heuristic": _policy_optimal_http,
    }

    results = {}
    for policy_name, policy_fn in policies.items():
        print(f"\n--- Running policy: {policy_name} ---")
        rewards_per_episode = []
        profits_per_episode = []

        n = max(1, num_episodes // len(policies))

        with StaffingAgencyEnv(base_url=env_url) as env:
            for ep in range(n):
                ep_seed = rng.randint(0, 99999)
                env.reset(seed=ep_seed)
                ep_reward = 0.0
                # Carry policy state across steps (for greedy/optimal)
                policy_state: dict = {}

                for _ in range(52):
                    action = policy_fn(env, rng, policy_state)
                    try:
                        result = env.step(action)
                        ep_reward += float(result.reward or 0.0)
                        if result.done:
                            break
                    except Exception:
                        pass

                # Read final profit from server state
                try:
                    fin = _req.post(
                        f"{env_url}/step",
                        json={"tool": "get_financial_summary", "params": {}},
                        timeout=5,
                    ).json()
                    obs_meta = fin.get("observation", {}).get("metadata", {})
                    tr = obs_meta.get("tool_result", {})
                    final_profit = tr.get("profit", 0.0)
                except Exception:
                    final_profit = 0.0

                rewards_per_episode.append(ep_reward)
                profits_per_episode.append(final_profit)

                if ep % max(1, n // 5) == 0:
                    print(f"  Episode {ep:3d}: reward={ep_reward:+8.3f}  profit=${final_profit:>10,.0f}")

        results[policy_name] = {
            "rewards": rewards_per_episode,
            "profits": profits_per_episode,
        }

    _plot_reward_curves(results, max(1, num_episodes // len(policies)))
    _save_metrics(results)

    print("\n[✓] Dry run complete. Reward curves saved to training/reward_curves.png")
    return results


# ---------------------------------------------------------------------------
# HTTP-based policies (work against the live server via client.py)
# ---------------------------------------------------------------------------

def _call(env, tool: str, params: dict = {}):
    """Helper: step the env with a StaffingAction, return tool_result dict."""
    from models import StaffingAction
    result = env.step(StaffingAction(tool=tool, params=params))
    tr = result.observation.tool_result if hasattr(result.observation, "tool_result") else {}
    return tr or {}, result


def _policy_random_http(env, rng, state: dict):
    from models import StaffingAction
    tools = [
        "find_available_projects", "get_agency_state",
        "get_candidate_state", "get_financial_summary",
        "find_candidate",
    ]
    return StaffingAction(tool=rng.choice(tools), params={})


def _policy_greedy_http(env, rng, state: dict):
    """Greedy: interview → hire → place. Reads state via GET tools each step."""
    from models import StaffingAction

    # Get candidate state
    cands_tr, _ = _call(env, "get_candidate_state")
    candidates = cands_tr.get("candidates", [])

    # Hire anyone in pipeline
    for c in candidates:
        if c.get("status") == "in_pipeline":
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # Place any hired candidate into a matching open role
    for c in candidates:
        if c.get("status") == "hired":
            projs_tr, _ = _call(env, "find_available_projects")
            for p in projs_tr.get("projects", []):
                for r in p.get("roles", []):
                    if not r.get("is_filled") and r.get("developer_type") == c.get("developer_type"):
                        return StaffingAction(
                            tool="match_candidate_to_project",
                            params={
                                "candidate_id": c["id"],
                                "project_id": p["project_id"],
                                "role_id": r["role_id"],
                            },
                        )

    # Interview first market candidate
    market_tr, _ = _call(env, "find_candidate")
    market = market_tr.get("candidates", [])
    if market:
        return StaffingAction(tool="interview_candidate", params={"candidate_id": market[0]["id"]})

    return StaffingAction(tool="get_agency_state", params={})


def _policy_optimal_http(env, rng, state: dict):
    """Demand-aware: only hire types with open roles, place immediately, pass expiring projects."""
    from models import StaffingAction

    projs_tr, _ = _call(env, "find_available_projects")
    projects = projs_tr.get("projects", [])

    # Build demand map and flag urgent projects
    demand: dict[str, int] = {}
    urgent_ids: list[str] = []
    for p in projects:
        for r in p.get("roles", []):
            if not r.get("is_filled"):
                dt = r.get("developer_type", "")
                demand[dt] = demand.get(dt, 0) + 1
        if p.get("deadline_remaining", 99) <= 3:
            urgent_ids.append(p["project_id"])

    cands_tr, _ = _call(env, "get_candidate_state")
    candidates = cands_tr.get("candidates", [])

    # Place hired candidates into matching roles first
    for c in candidates:
        if c.get("status") == "hired":
            for p in projects:
                for r in p.get("roles", []):
                    if not r.get("is_filled") and r.get("developer_type") == c.get("developer_type"):
                        return StaffingAction(
                            tool="match_candidate_to_project",
                            params={
                                "candidate_id": c["id"],
                                "project_id": p["project_id"],
                                "role_id": r["role_id"],
                            },
                        )

    # Hire pipeline candidates whose type is in demand
    for c in candidates:
        if c.get("status") == "in_pipeline" and c.get("developer_type") in demand:
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # Interview market candidates matching demand
    if demand:
        for dt in demand:
            market_tr, _ = _call(env, "find_candidate", {"developer_type": dt})
            market = market_tr.get("candidates", [])
            if market:
                return StaffingAction(tool="interview_candidate", params={"candidate_id": market[0]["id"]})

    # Pass on urgent projects we can't fill
    hired_types = {c.get("developer_type") for c in candidates if c.get("status") == "hired"}
    for p in projects:
        if p["project_id"] in urgent_ids:
            needs = {r.get("developer_type") for r in p.get("roles", []) if not r.get("is_filled")}
            if not needs.intersection(hired_types):
                return StaffingAction(tool="pass_on_project", params={"project_id": p["project_id"]})

    return StaffingAction(tool="get_financial_summary", params={})


def _plot_reward_curves(results: dict, n_episodes: int):
    """Plot and save reward curves for all policies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        Path("training").mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {"random": "#e74c3c", "greedy": "#f39c12", "optimal_heuristic": "#27ae60"}

        for policy_name, data in results.items():
            rewards = data["rewards"]
            profits = data["profits"]
            color = colors.get(policy_name, "blue")
            x = range(len(rewards))

            # Smoothed with rolling window
            window = max(1, len(rewards) // 10)
            smoothed_r = np.convolve(rewards, np.ones(window)/window, mode="valid")
            smoothed_p = np.convolve(profits, np.ones(window)/window, mode="valid")
            x_smooth = range(len(smoothed_r))

            axes[0].plot(x, rewards, alpha=0.2, color=color)
            axes[0].plot(x_smooth, smoothed_r, color=color, linewidth=2, label=policy_name)

            axes[1].plot(x, profits, alpha=0.2, color=color)
            axes[1].plot(x_smooth, smoothed_p, color=color, linewidth=2, label=policy_name)

        axes[0].set_title("Episode Reward (scaled) — Before/After Policy Comparison", fontsize=12)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Episode Profit ($) — Before/After Policy Comparison", fontsize=12)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Net Profit ($)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        plt.suptitle(
            "Staffing Agency RL — Policy Comparison\n"
            "random → greedy → optimal heuristic (simulates training progression)",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig("training/reward_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[✓] Saved: training/reward_curves.png")
    except ImportError:
        print("[!] matplotlib not installed — skipping plot. Install with: uv pip install matplotlib")


def _save_metrics(results: dict):
    """Save raw metrics to JSON for later analysis."""
    import json
    Path("training").mkdir(exist_ok=True)
    summary = {}
    for policy, data in results.items():
        rewards = data["rewards"]
        profits = data["profits"]
        summary[policy] = {
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "mean_profit": round(sum(profits) / len(profits), 2),
            "max_profit": round(max(profits), 2),
            "min_profit": round(min(profits), 2),
            "positive_profit_rate": round(sum(1 for p in profits if p > 0) / len(profits), 3),
        }
    with open("training/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[✓] Saved: training/metrics_summary.json")
    print("\nMetrics Summary:")
    for policy, m in summary.items():
        print(f"  {policy:25s}: mean_profit=${m['mean_profit']:>10,.0f}  "
              f"positive_rate={m['positive_profit_rate']:.1%}")


# ---------------------------------------------------------------------------
# Full TRL GRPO training (requires GPU + model)
# ---------------------------------------------------------------------------

def train_grpo(args):
    """Full GRPO training loop using TRL."""
    try:
        from trl import GRPOTrainer, GRPOConfig
        from trl.experimental.openenv import generate_rollout_completions
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("[!] TRL/transformers not installed. Run with --dry_run or:")
        print("    uv pip install -e '.[train]'")
        sys.exit(1)

    from client import StaffingAgencyEnv

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    env_client = StaffingAgencyEnv(base_url=args.env_url).sync()

    def rollout_func(prompts: list[str], trainer: GRPOTrainer):
        """OpenEnv rollout function for GRPOTrainer."""
        outputs = generate_rollout_completions(trainer, prompts)
        tok = trainer.processing_class

        env_rewards = []
        format_rewards = []

        for out in outputs:
            completion_text = tok.decode(out["completion_ids"], skip_special_tokens=True)
            parsed = parse_tool_call(completion_text)
            fmt_r = reward_fn_tool_format([completion_text])[0]
            format_rewards.append(fmt_r)

            if parsed:
                tool_name, params = parsed
                try:
                    from openenv.core.env_server.mcp_types import CallToolAction
                    result = env_client.step(CallToolAction(tool_name=tool_name, arguments=params))
                    env_rewards.append(float(result.reward or 0.0) + fmt_r)
                except Exception:
                    env_rewards.append(fmt_r - 0.1)
            else:
                env_rewards.append(fmt_r)

        return {
            "prompt_ids": [o["prompt_ids"] for o in outputs],
            "completion_ids": [o["completion_ids"] for o in outputs],
            "logprobs": [o["logprobs"] for o in outputs],
            "env_reward": env_rewards,
        }

    # Reset env at episode boundaries
    env_client.reset(seed=args.seed)

    # Build dataset (system + initial observation prompts)
    from datasets import Dataset
    obs = env_client.reset(seed=args.seed)
    initial_state = str(obs)

    dataset = Dataset.from_dict({
        "prompt": [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current state:\n{initial_state}\nWhat is your action?"},
                ],
                tokenize=False, add_generation_prompt=True,
            )
        ] * args.num_episodes
    })

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=256,
        reward_funcs=[reward_fn_profit, reward_fn_tool_format, reward_fn_placement],
        logging_steps=10,
        save_steps=50,
        seed=args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn_profit, reward_fn_tool_format, reward_fn_placement],
        rollout_function=rollout_func,
    )

    print(f"\nStarting GRPO training: {args.num_episodes} episodes")
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_url}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n[✓] Training complete. Model saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    Path("training").mkdir(exist_ok=True)

    if args.dry_run:
        dry_run_simulate(
            env_url=args.env_url,
            num_episodes=args.num_episodes,
            max_turns=args.max_turns,
            seed=args.seed,
        )
    else:
        train_grpo(args)
