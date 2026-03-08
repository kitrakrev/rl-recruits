"""
Dry-run simulator — validate the full HTTP stack without a GPU.

Uses the three heuristic policies to run episodes against the live server and
produces reward curves showing random → greedy → optimal improvement.

Run:
    uv run python training/train_grpo.py --dry_run --num_episodes 90
"""
from __future__ import annotations

import random
import sys

from training.metrics import plot_reward_curves, save_metrics
from training.policies import policy_random, policy_greedy, policy_optimal


def dry_run_simulate(
    env_url: str,
    num_episodes: int,
    max_turns: int,
    seed: int,
) -> dict:
    """
    Simulate training without a model by running three heuristic policies.

    Hits the live server at env_url — make sure it is running:
        uvicorn server.app:app --host 0.0.0.0 --port 8000

    Parameters
    ----------
    env_url      : base URL of the environment server
    num_episodes : total episodes split evenly across the three policies
    max_turns    : steps per episode (each step = one tool call)
    seed         : random seed

    Returns
    -------
    dict: {policy_name: {"rewards": [...], "profits": [...]}}
    """
    import requests as _req

    print("\n" + "=" * 60)
    print("DRY RUN MODE — Simulating reward curves via live server")
    print(f"Server: {env_url}")
    print("=" * 60)

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

    _POLICIES = {
        "random":            policy_random,
        "greedy":            policy_greedy,
        "optimal_heuristic": policy_optimal,
    }

    results: dict = {}
    n_per_policy = max(1, num_episodes // len(_POLICIES))

    for policy_name, policy_fn in _POLICIES.items():
        print(f"\n--- Running policy: {policy_name} ({n_per_policy} episodes) ---")
        rewards_per_episode: list[float] = []
        profits_per_episode: list[float] = []

        with StaffingAgencyEnv(base_url=env_url) as env:
            for ep in range(n_per_policy):
                ep_seed = rng.randint(0, 99_999)
                env.reset(seed=ep_seed)
                ep_reward = 0.0
                final_profit = 0.0
                policy_state: dict = {
                    "_step": 0,
                    "_last_proj_refresh": -99,
                    "_last_mkt_refresh":  -99,
                }

                for tick in range(max_turns):
                    policy_state["_step"] = policy_state.get("_step", 0) + 1
                    action = policy_fn(env, rng, policy_state)
                    try:
                        result = env.step(action)
                        ep_reward   += float(result.reward or 0.0)
                        final_profit = result.observation.profit or 0.0

                        # Update cache from tool result
                        tr = result.observation.tool_result or {}
                        if isinstance(tr, dict):
                            _update_policy_state(policy_state, action.tool, tr)

                        if result.done:
                            break
                    except Exception:
                        pass

                # Confirm final profit via get_financial_summary
                try:
                    fin = env.step(StaffingAction(tool="get_financial_summary", params={}))
                    final_profit = (fin.observation.tool_result or {}).get("profit", final_profit)
                except Exception:
                    pass

                rewards_per_episode.append(ep_reward)
                profits_per_episode.append(final_profit)

                if ep % max(1, n_per_policy // 5) == 0:
                    print(f"  Episode {ep:3d}: reward={ep_reward:+8.3f}  profit=${final_profit:>10,.0f}")

        results[policy_name] = {
            "rewards": rewards_per_episode,
            "profits": profits_per_episode,
        }

    plot_reward_curves(results, n_per_policy)
    save_metrics(results)
    print("\n[✓] Dry run complete. Reward curves saved to training/reward_curves.png")
    return results


def _update_policy_state(state: dict, tool: str, tr: dict) -> None:
    """Update the policy's cache dict from a tool result."""
    if tool == "find_candidate":
        state["market"] = tr.get("candidates", [])
        state["_last_mkt_refresh"] = state["_step"]
    elif tool == "find_available_projects":
        state["projects"] = tr.get("projects", [])
        state["_last_proj_refresh"] = state["_step"]
    elif tool == "interview_candidate" and tr.get("success"):
        cid = tr.get("candidate_id", "")
        interviewed = None
        new_market = []
        for mc in state.get("market", []):
            if mc.get("id") == cid:
                interviewed = mc
            else:
                new_market.append(mc)
        state["market"] = new_market
        if interviewed:
            interviewed["status"] = "in_pipeline"
            interviewed["base_rating"] = tr.get("base_rating", 0)
            state.setdefault("candidates", []).append(interviewed)
    elif tool == "interview_candidate" and not tr.get("success"):
        cid = tr.get("candidate_id", "") or ""
        state["market"] = [mc for mc in state.get("market", []) if mc.get("id") != cid]
    elif tool == "hire_candidate" and tr.get("success"):
        cid = tr.get("candidate_id", "")
        for c in state.get("candidates", []):
            if c.get("id") == cid:
                c["status"] = "hired"
                c["salary_weekly"] = tr.get("salary_weekly", 0)
                c["composite_rating"] = tr.get("composite_rating", 0)
                break
    elif tool == "match_candidate_to_project":
        cid = tr.get("candidate_id", "")
        rid = tr.get("role_id", "")
        pid = tr.get("project_id", "")
        if tr.get("success"):
            for c in state.get("candidates", []):
                if c.get("id") == cid:
                    c["status"] = "placed"
                    break
            for p in state.get("projects", []):
                for r in p.get("roles", []):
                    if r.get("role_id") == rid:
                        r["filled_count"] = r.get("filled_count", 0) + 1
                        if r["filled_count"] >= r.get("headcount", 1):
                            r["is_filled"] = True
        elif "not found" in str(tr.get("error", "")).lower():
            state["projects"] = [p for p in state.get("projects", []) if p.get("project_id") != pid]
        else:
            state.setdefault("_failed_matches", set()).add((cid, rid))
    elif tool == "let_go_candidate" and tr.get("success"):
        cid = tr.get("candidate_id", "")
        state["candidates"] = [c for c in state.get("candidates", []) if c.get("id") != cid]
    elif tool == "pass_on_project" and tr.get("success"):
        pid = tr.get("project_id", "")
        state["projects"] = [p for p in state.get("projects", []) if p.get("project_id") != pid]
