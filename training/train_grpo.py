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
import random
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

CRITICAL ECONOMICS (updated):
- Interview costs $500 per candidate screened — be selective
- Salary is DYNAMIC: every candidate has a unique salary_expectation (their floor)
  → Junior backend ~$75k/yr, Senior ML Engineer ~$150k × 1.3 × skill modifier
- Bill rate is VARIABLE per role: what the client pays (set by project, $130k–$300k+/yr)
- TRUE MARGIN = bill_rate_weekly − salary_weekly per placed candidate per week
  → A cheap junior (salary $1,200/wk) on a $3,000/wk bill-rate role = +$1,800/wk profit
  → An expensive senior ($3,000/wk) on a $2,800/wk role = −$200/wk LOSS every week
- Benched candidate = −salary_weekly BURN per week (dynamic, not fixed)
- Onboarding: −$2,000 one-time per hire
- Projects must be FULLY filled (SEALED) to lock in recurring revenue
- Expired projects = large penalty; client churn (satisfaction < 0.3) = $50,000 LTV loss

STRATEGY HINTS:
- Use negotiate_salary to lower a candidate's salary before hiring them
- Check bill_rate_weekly on each role before committing a candidate — only place where margin > 0
- Use get_market_demand to identify which developer types are most needed
- Confirm projects before committing candidates (confirm_project)
- Pass on projects where you cannot fill all roles — preventing expiry avoids the penalty
- Let go of benched candidates whose salary exceeds any available bill rate (they lose money)

Use the available tools to manage your agency. Think step by step.
When you want to call a tool, output EXACTLY:
TOOL: <tool_name>
PARAMS: <json_params_or_empty_dict>

Example:
TOOL: get_agency_state
PARAMS: {}

TOOL: find_candidate
PARAMS: {"developer_type": "backend"}

TOOL: negotiate_salary
PARAMS: {"candidate_id": "C-BA-abc12345", "offer_weekly": 1400.0}

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
    print("Curriculum: stage 1→2→3 across policy runs (easier→harder)")
    print("  random=stage1, greedy=stage2, optimal=stage3\n")

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
                final_profit = 0.0
                # policy_state caches candidate/project lists fetched this step
                policy_state: dict = {"_step": 0, "_last_proj_refresh": -99, "_last_mkt_refresh": -99}
                verbose = False  # set True for debug: (ep == 0 and policy_name == "optimal_heuristic")

                for tick in range(52):
                    policy_state["_step"] = policy_state.get("_step", 0) + 1
                    action = policy_fn(env, rng, policy_state)
                    try:
                        result = env.step(action)
                        ep_reward += float(result.reward or 0.0)
                        final_profit = result.observation.profit
                        if verbose:
                            tr_dbg = result.observation.tool_result or {}
                            succ = tr_dbg.get("success", "n/a") if isinstance(tr_dbg, dict) else "n/a"
                            err = tr_dbg.get("error", "")[:60] if isinstance(tr_dbg, dict) else ""
                            n_cands = len(policy_state.get("candidates", []))
                            n_hired = sum(1 for c in policy_state.get("candidates", []) if c.get("status") == "hired")
                            n_pipe = sum(1 for c in policy_state.get("candidates", []) if c.get("status") == "in_pipeline")
                            n_placed = sum(1 for c in policy_state.get("candidates", []) if c.get("status") == "placed")
                            print(f"    T{tick:2d} {action.tool:30s} ok={str(succ):5s} rwd={float(result.reward or 0):+7.2f} "
                                  f"pft=${final_profit:>8,.0f} pipe={n_pipe} hire={n_hired} plcd={n_placed} "
                                  f"{err}")
                        # Update cache for next step's policy decision
                        tr = result.observation.tool_result or {}
                        if isinstance(tr, dict):
                            if action.tool == "find_candidate":
                                policy_state["market"] = tr.get("candidates", [])
                                policy_state["_last_mkt_refresh"] = policy_state["_step"]
                            elif action.tool == "find_available_projects":
                                policy_state["projects"] = tr.get("projects", [])
                                policy_state["_last_proj_refresh"] = policy_state["_step"]
                            elif action.tool == "interview_candidate" and tr.get("success"):
                                # Move candidate from market cache → candidates (pipeline)
                                cid = tr.get("candidate_id", "")
                                interviewed = None
                                new_market = []
                                for mc in policy_state.get("market", []):
                                    if mc.get("id") == cid:
                                        interviewed = mc
                                    else:
                                        new_market.append(mc)
                                policy_state["market"] = new_market
                                if interviewed:
                                    interviewed["status"] = "in_pipeline"
                                    interviewed["base_rating"] = tr.get("base_rating", 0)
                                    policy_state.setdefault("candidates", []).append(interviewed)
                            elif action.tool == "hire_candidate" and tr.get("success"):
                                # Update candidate status: in_pipeline → hired
                                cid = tr.get("candidate_id", "")
                                for c in policy_state.get("candidates", []):
                                    if c.get("id") == cid:
                                        c["status"] = "hired"
                                        c["salary_weekly"] = tr.get("salary_weekly", 0)
                                        c["composite_rating"] = tr.get("composite_rating", 0)
                                        break
                            elif action.tool == "match_candidate_to_project":
                                cid = action.params.get("candidate_id", "")
                                rid = action.params.get("role_id", "")
                                pid = action.params.get("project_id", "")
                                if tr.get("success"):
                                    # Update candidate status: hired → placed
                                    for c in policy_state.get("candidates", []):
                                        if c.get("id") == cid:
                                            c["status"] = "placed"
                                            break
                                    # Mark role as filled in cached projects
                                    for p in policy_state.get("projects", []):
                                        for r in p.get("roles", []):
                                            if r.get("role_id") == rid:
                                                r["filled_count"] = r.get("filled_count", 0) + 1
                                                if r["filled_count"] >= r.get("headcount", 1):
                                                    r["is_filled"] = True
                                else:
                                    err = tr.get("error", "")
                                    if "not found" in err.lower():
                                        # Project or role expired — remove stale cache entry
                                        policy_state["projects"] = [
                                            p for p in policy_state.get("projects", [])
                                            if p.get("project_id") != pid
                                        ]
                                    else:
                                        # Track failed match to avoid retrying
                                        policy_state.setdefault("_failed_matches", set()).add((cid, rid))
                            elif action.tool == "interview_candidate" and not tr.get("success"):
                                # Candidate left market — remove from cache
                                cid_fail = tr.get("candidate_id", "") or action.params.get("candidate_id", "")
                                policy_state["market"] = [
                                    mc for mc in policy_state.get("market", [])
                                    if mc.get("id") != cid_fail
                                ]
                            elif action.tool == "let_go_candidate" and tr.get("success"):
                                cid = tr.get("candidate_id", "")
                                policy_state["candidates"] = [
                                    c for c in policy_state.get("candidates", [])
                                    if c.get("id") != cid
                                ]
                            elif action.tool == "pass_on_project" and tr.get("success"):
                                pid = action.params.get("project_id", "")
                                policy_state["projects"] = [
                                    p for p in policy_state.get("projects", [])
                                    if p.get("project_id") != pid
                                ]
                        if result.done:
                            break
                    except Exception:
                        pass

                # Confirm final profit via get_financial_summary (GET tool, no tick)
                try:
                    fin = env.step(StaffingAction(tool="get_financial_summary", params={}))
                    final_profit = (fin.observation.tool_result or {}).get("profit", final_profit)
                except Exception:
                    pass  # fall back to last seen profit from loop

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
#
# IMPORTANT: Every env.step() call advances the episode (for EXECUTE tools).
# Policies must NOT make extra env.step() calls to "read" state — that wastes
# episode steps and gives 0 reward. Instead they use policy_state (a dict cache
# populated by the previous step's tool_result) to make decisions.
# ---------------------------------------------------------------------------

def _policy_random_http(env, rng, state: dict):
    """Random: basic pipeline with random candidate/project selection."""
    from models import StaffingAction

    candidates = state.get("candidates", [])
    projects   = state.get("projects", [])
    market     = state.get("market", [])

    # Hire anyone in pipeline first
    for c in candidates:
        if c.get("status") == "in_pipeline":
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # Try to place a hired candidate into any open role
    hired = [c for c in candidates if c.get("status") == "hired"]
    if hired and projects:
        c = rng.choice(hired)
        for p in projects:
            for r in p.get("roles", []):
                open_role = not r.get("is_filled") if "is_filled" in r else r.get("filled_count", 0) < r.get("headcount", 1)
                if open_role:
                    return StaffingAction(
                        tool="match_candidate_to_project",
                        params={"candidate_id": c["id"], "project_id": p["project_id"], "role_id": r["role_id"]},
                    )

    # Interview a random market candidate
    if market:
        c = rng.choice(market)
        return StaffingAction(tool="interview_candidate", params={"candidate_id": c["id"]})

    # Refresh caches
    execute_tools = ["find_available_projects", "find_candidate"]
    return StaffingAction(tool=rng.choice(execute_tools), params={})


def _policy_greedy_http(env, rng, state: dict):
    """
    Greedy: interview → hire → place pipeline.
    Uses cached candidates/projects from policy_state (populated last step).
    Checks developer_type, skill_score, and seniority compatibility before matching.
    """
    from models import StaffingAction

    candidates = state.get("candidates", [])
    projects   = state.get("projects", [])
    market     = state.get("market", [])
    failed_matches = state.setdefault("_failed_matches", set())

    # Adjacency: which candidate types can fill which role types
    ADJACENT = {
        "backend":     {"backend", "fullstack", "ml_engineer"},
        "frontend":    {"frontend", "fullstack"},
        "fullstack":   {"fullstack", "backend", "frontend"},
        "ml_engineer": {"ml_engineer", "backend"},
        "devops":      {"devops"},
    }
    SEN_ORDER = {"junior": 0, "mid": 1, "senior": 2}

    def role_is_open(r: dict) -> bool:
        """Check if role is unfilled (handles missing is_filled key)."""
        if "is_filled" in r:
            return not r["is_filled"]
        return r.get("filled_count", 0) < r.get("headcount", 1)

    def can_fill(cand: dict, role: dict) -> bool:
        """Full compatibility: type + skill + seniority."""
        cand_type = cand.get("developer_type", "")
        role_type = role.get("developer_type", "")
        if role_type not in ADJACENT.get(cand_type, {cand_type}):
            return False
        if cand.get("skill_score", 0) < role.get("min_skill_score", 0):
            return False
        cand_sen = SEN_ORDER.get(cand.get("seniority_level", "junior"), 0)
        role_sen = SEN_ORDER.get(role.get("seniority", "junior"), 0)
        if cand_sen < role_sen:
            return False
        return True

    def can_fill_type(cand_type: str, role_type: str) -> bool:
        return role_type in ADJACENT.get(cand_type, {cand_type})

    hired = [c for c in candidates if c.get("status") == "hired"]
    pipeline = [c for c in candidates if c.get("status") == "in_pipeline"]
    all_available = hired + pipeline + market

    open_roles = [
        (p, r) for p in projects for r in p.get("roles", [])
        if role_is_open(r)
    ]

    # 0. Proactively PASS on unfillable projects (no satisfaction penalty)
    step = state.get("_step", 0)
    stale_offset = max(0, step - state.get("_last_proj_refresh", 0))
    for p in projects:
        open_roles_p = [r for r in p.get("roles", []) if role_is_open(r)]
        if not open_roles_p:
            continue
        fillable = all(
            any(can_fill(c, r) for c in all_available)
            for r in open_roles_p
        )
        deadline_est = max(0, p.get("deadline_remaining", 99) - stale_offset)
        if not fillable or (deadline_est <= 3 and not any(
            can_fill(c, r) for c in hired for r in open_roles_p
        )):
            return StaffingAction(tool="pass_on_project", params={"project_id": p["project_id"]})

    # 1. Hire anyone in pipeline
    for c in pipeline:
        return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # 2. Place hired candidate into first compatible open role (skip failed pairs)
    for c in hired:
        for p, r in open_roles:
            pair_key = (c["id"], r["role_id"])
            if can_fill(c, r) and pair_key not in failed_matches:
                return StaffingAction(
                    tool="match_candidate_to_project",
                    params={
                        "candidate_id": c["id"],
                        "project_id":   p["project_id"],
                        "role_id":      r["role_id"],
                    },
                )

    # 3. Let go of bench candidates with no matching open roles (stop burn)
    for c in hired:
        placeable = any(
            can_fill(c, r) and (c["id"], r["role_id"]) not in failed_matches
            for _p, r in open_roles
        )
        if not placeable:
            return StaffingAction(tool="let_go_candidate", params={"candidate_id": c["id"]})

    # 4. Interview market candidate whose type matches an open role
    for mc in market:
        for _p, r in open_roles:
            if can_fill(mc, r):
                return StaffingAction(tool="interview_candidate", params={"candidate_id": mc["id"]})

    # 5. Refresh stale caches — ALTERNATE between projects and market
    step = state.get("_step", 0)
    proj_age = step - state.get("_last_proj_refresh", -99)
    mkt_age = step - state.get("_last_mkt_refresh", -99)
    # Always alternate: never call the same refresh twice in a row
    last_refresh = state.get("_last_refresh_tool", "")
    if last_refresh == "find_available_projects":
        state["_last_refresh_tool"] = "find_candidate"
        return StaffingAction(tool="find_candidate", params={})
    if last_refresh == "find_candidate":
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    # First refresh — prefer whichever is older
    if proj_age >= mkt_age:
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    state["_last_refresh_tool"] = "find_candidate"
    return StaffingAction(tool="find_candidate", params={})


def _policy_optimal_http(env, rng, state: dict):
    """
    Demand-aware optimal: target easiest-to-seal projects (fewest unfilled roles),
    only hire types that match open roles, place immediately, pass expiring unwinnable projects.
    Full skill/seniority compatibility checks to avoid wasted match attempts.
    """
    from models import StaffingAction

    candidates = state.get("candidates", [])
    projects   = state.get("projects", [])
    market     = state.get("market", [])
    failed_matches = state.setdefault("_failed_matches", set())

    ADJACENT = {
        "backend":     {"backend", "fullstack", "ml_engineer"},
        "frontend":    {"frontend", "fullstack"},
        "fullstack":   {"fullstack", "backend", "frontend"},
        "ml_engineer": {"ml_engineer", "backend"},
        "devops":      {"devops"},
    }
    SEN_ORDER = {"junior": 0, "mid": 1, "senior": 2}

    def role_is_open(r: dict) -> bool:
        if "is_filled" in r:
            return not r["is_filled"]
        return r.get("filled_count", 0) < r.get("headcount", 1)

    def can_fill(cand: dict, role: dict) -> bool:
        """Full compatibility: type + skill + seniority."""
        cand_type = cand.get("developer_type", "")
        role_type = role.get("developer_type", "")
        if role_type not in ADJACENT.get(cand_type, {cand_type}):
            return False
        if cand.get("skill_score", 0) < role.get("min_skill_score", 0):
            return False
        cand_sen = SEN_ORDER.get(cand.get("seniority_level", "junior"), 0)
        role_sen = SEN_ORDER.get(role.get("seniority", "junior"), 0)
        if cand_sen < role_sen:
            return False
        return True

    def can_fill_type(cand_type: str, role_type: str) -> bool:
        return role_type in ADJACENT.get(cand_type, {cand_type})

    # Estimate deadline staleness: cached deadline - steps since last project refresh
    step = state.get("_step", 0)
    proj_refresh_step = state.get("_last_proj_refresh", 0)
    stale_offset = max(0, step - proj_refresh_step)

    def est_deadline(p: dict) -> int:
        return max(0, p.get("deadline_remaining", 99) - stale_offset)

    # Sort projects: most deadline remaining first (safest to fill)
    sorted_projects = sorted(projects, key=est_deadline, reverse=True)

    # Build demand from open roles
    open_roles_all = []
    demand: dict[str, int] = {}
    for p in sorted_projects:
        for r in p.get("roles", []):
            if role_is_open(r):
                dt = r.get("developer_type", "")
                demand[dt] = demand.get(dt, 0) + 1
                open_roles_all.append((p, r))

    hired = [c for c in candidates if c.get("status") == "hired"]
    pipeline = [c for c in candidates if c.get("status") == "in_pipeline"]
    all_available = hired + pipeline + market  # all candidates we might use

    # 0. Proactively PASS on unfillable projects (prevents expiry penalties + churn)
    #    pass_on_project has NO satisfaction penalty — it just removes the project
    for p in list(sorted_projects):
        open_roles_p = [r for r in p.get("roles", []) if role_is_open(r)]
        if not open_roles_p:
            continue
        # Can any available candidate (hired, pipeline, market) fill each open role?
        fillable = all(
            any(can_fill(c, r) for c in all_available)
            for r in open_roles_p
        )
        deadline_est = est_deadline(p)
        # Pass if: no compatible candidate exists, OR deadline too tight to pipeline
        if not fillable or (deadline_est <= 3 and not any(
            can_fill(c, r) for c in hired for r in open_roles_p
        )):
            return StaffingAction(tool="pass_on_project", params={"project_id": p["project_id"]})

    # 1. Place any hired candidate into best matching open role
    for p, r in open_roles_all:
        for c in hired:
            pair_key = (c["id"], r["role_id"])
            if can_fill(c, r) and pair_key not in failed_matches:
                return StaffingAction(
                    tool="match_candidate_to_project",
                    params={
                        "candidate_id": c["id"],
                        "project_id":   p["project_id"],
                        "role_id":      r["role_id"],
                    },
                )

    # 2. Hire pipeline candidates whose type is needed
    for c in pipeline:
        if any(can_fill_type(c.get("developer_type", ""), dt) for dt in demand):
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # 3. Let go of bench candidates with no matching open roles (stop burn)
    for c in hired:
        placeable = any(
            can_fill(c, r) and (c["id"], r["role_id"]) not in failed_matches
            for _p, r in open_roles_all
        )
        if not placeable:
            return StaffingAction(tool="let_go_candidate", params={"candidate_id": c["id"]})

    # 5. Interview market candidate matching demand (highest skill first)
    market_sorted = sorted(market, key=lambda c: c.get("skill_score", 0), reverse=True)
    for mc in market_sorted:
        for _p, r in open_roles_all:
            if can_fill(mc, r):
                return StaffingAction(tool="interview_candidate", params={"candidate_id": mc["id"]})

    # 6. Refresh stale caches — ALTERNATE between projects and market
    step = state.get("_step", 0)
    proj_age = step - state.get("_last_proj_refresh", -99)
    mkt_age = step - state.get("_last_mkt_refresh", -99)
    last_refresh = state.get("_last_refresh_tool", "")
    if last_refresh == "find_available_projects":
        state["_last_refresh_tool"] = "find_candidate"
        return StaffingAction(tool="find_candidate", params={})
    if last_refresh == "find_candidate":
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    if proj_age >= mkt_age:
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    state["_last_refresh_tool"] = "find_candidate"
    return StaffingAction(tool="find_candidate", params={})


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

        # Title adapts to whether this is a dry-run policy comparison or a
        # live GRPO training run with the iterative Monte Carlo loop.
        is_grpo = "grpo_mc_training" in results
        if is_grpo:
            suptitle = (
                "Staffing Agency RL — GRPO Monte Carlo Training\n"
                "Iterative loop: rollout 52 weeks → assign episode profit → 1-epoch GRPO update"
            )
        else:
            suptitle = (
                "Staffing Agency RL — Policy Comparison\n"
                "random → greedy → optimal heuristic (simulates training progression)"
            )
        plt.suptitle(suptitle, fontsize=11)
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
# Full TRL GRPO training — Iterative Monte Carlo Rollout Loop
#
# The core insight: TRL's GRPOTrainer is a single-turn prompt→completion
# optimizer.  If we hand it a static dataset of Week-1 prompts and run it
# end-to-end, it only ever optimises the FIRST action in the 52-step game —
# the other 51 weeks are invisible to the gradient.
#
# The fix: an outer ITERATIVE loop.
#   1. ROLLOUT PHASE  — Use the CURRENT model to play N full 52-week episodes
#      against the live StaffingAgencyEnv.  Record every (prompt, completion)
#      pair along with the episode's final profit.
#   2. DATASET PHASE  — Assign the final episode profit as the reward for EVERY
#      step in the trajectory (Monte Carlo return: all actions in a good episode
#      get positive signal; all actions in a bad episode get negative signal).
#   3. TRAIN PHASE    — Construct a fresh Dataset from this trajectory and run
#      GRPOTrainer for 1 epoch.  The model sees the full 52-step context.
#   4. REPEAT         — Loop back to Phase 1.  As the model improves, profit rises.
#
# Graphs: we hook directly into the outer loop's final_profit list, so the
# "Profit per Episode" curve reflects true 52-week episode outcomes — not the
# reward from a single tool call.
# ---------------------------------------------------------------------------

def train_grpo(args):
    """Iterative Monte Carlo GRPO training loop using TRL."""
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import requests as _req
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: uv pip install -e '.[train]'")
        sys.exit(1)

    from client import StaffingAgencyEnv
    from models import StaffingAction
    from datasets import Dataset

    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # -----------------------------------------------------------------------
    # Rollout: play one full 52-step episode, return (prompts, completions, profit)
    # -----------------------------------------------------------------------
    def rollout_full_episode(env: "StaffingAgencyEnv", seed: int) -> tuple[list[str], list[str], float]:
        """
        Run one complete 52-week episode.

        Returns:
          prompts      — list of prompt strings (one per week step)
          completions  — list of model completion strings (one per week step)
          final_profit — episode-end net profit ($)
        """
        env.reset(seed=seed)
        prompts_out: list[str] = []
        completions_out: list[str] = []
        final_profit = 0.0

        # Seed the conversation with initial state observation
        obs_text = "Episode started. Week 1 of 52."
        try:
            fin = env.step(StaffingAction(tool="get_financial_summary", params={}))
            obs_text = json.dumps((fin.observation.tool_result or {}), indent=2)
            final_profit = float((fin.observation.tool_result or {}).get("profit", 0.0))
        except Exception:
            pass

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Week 1 of 52. Current financials:\n{obs_text}\nWhat is your action?"},
        ]

        for week in range(1, 53):
            # Build prompt string for this step
            prompt_str = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Generate one completion from the model
            input_ids = tokenizer(
                prompt_str, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Extract only the newly generated tokens
            completion_ids = out_ids[0][input_ids.shape[-1]:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Execute the tool call in the environment
            parsed = parse_tool_call(completion_text)
            reward = 0.0
            tool_result_text = "No tool parsed — no action taken."
            done = (week == 52)

            if parsed:
                tool_name, params = parsed
                try:
                    result = env.step(StaffingAction(tool=tool_name, params=params))
                    reward = float(result.reward or 0.0)
                    tr = result.observation.tool_result or {}
                    tool_result_text = json.dumps(tr, indent=2)[:400]
                    # Track latest profit from every get_financial_summary call
                    if isinstance(tr, dict) and "profit" in tr:
                        final_profit = float(tr["profit"])
                    done = result.done or done
                except Exception as e:
                    tool_result_text = f"Error: {e}"

            prompts_out.append(prompt_str)
            completions_out.append(completion_text)

            if done:
                break

            # Update conversation with result
            conversation.append({"role": "assistant", "content": completion_text})
            conversation.append({
                "role": "user",
                "content": (
                    f"Week {week + 1} of 52. Tool result:\n{tool_result_text}\n"
                    f"Step reward: {reward:+.2f}\n"
                    "What is your next action?"
                ),
            })

        # Final profit snapshot
        try:
            fin = env.step(StaffingAction(tool="get_financial_summary", params={}))
            final_profit = float((fin.observation.tool_result or {}).get("profit", final_profit))
        except Exception:
            pass

        return prompts_out, completions_out, final_profit

    # -----------------------------------------------------------------------
    # Reward functions used by GRPOTrainer (single-step, called per completion)
    # They receive the Monte Carlo return injected via the dataset column.
    # -----------------------------------------------------------------------
    def reward_fn_mc_profit(completions: list[str], mc_return: list[float] = None, **kwargs) -> list[float]:
        """Monte Carlo return: final episode profit assigned to every step."""
        if mc_return:
            return [float(r) / 1_000.0 for r in mc_return]   # scale for stability
        return [0.0] * len(completions)

    def reward_fn_tool_format(completions: list[str], **kwargs) -> list[float]:
        """Shaping: +0.1 for valid TOOL:/PARAMS: format."""
        rewards = []
        for c in completions:
            if "TOOL:" in c and "PARAMS:" in c:
                rewards.append(0.1 if parse_tool_call(c) else -0.05)
            else:
                rewards.append(-0.1)
        return rewards

    def reward_fn_placement(completions: list[str], **kwargs) -> list[float]:
        """Shaping: bonus for high-value action types."""
        rewards = []
        for c in completions:
            parsed = parse_tool_call(c)
            if parsed and parsed[0] == "match_candidate_to_project":
                rewards.append(0.2)
            elif parsed and parsed[0] in ("hire_candidate", "negotiate_salary"):
                rewards.append(0.05)
            else:
                rewards.append(0.0)
        return rewards

    # -----------------------------------------------------------------------
    # Training config — 1 epoch per outer iteration (fresh dataset each time)
    # -----------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        learning_rate=5e-6,
        max_completion_length=128,
        logging_steps=1,
        save_steps=50,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Outer iterative loop
    # -----------------------------------------------------------------------
    rng = random.Random(args.seed)
    all_profits: list[float] = []          # one entry per episode, for graphing
    all_rewards: list[float] = []

    print(f"\nStarting iterative GRPO training")
    print(f"  model:    {args.model_name}")
    print(f"  env:      {args.env_url}")
    print(f"  episodes: {args.num_episodes}  (each = 52-step rollout + 1-epoch train)")

    # Verify env is up
    try:
        import requests as _req
        _req.get(f"{args.env_url}/health", timeout=5).raise_for_status()
        print("[✓] Environment server is healthy\n")
    except Exception as e:
        print(f"[✗] Server not reachable at {args.env_url}: {e}")
        print("    Start it with: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    with StaffingAgencyEnv(base_url=args.env_url) as env:
        for ep in range(args.num_episodes):
            ep_seed = rng.randint(0, 99_999)

            # ---- PHASE 1: Rollout ----------------------------------------
            prompts, completions, final_profit = rollout_full_episode(env, ep_seed)

            # ---- PHASE 2: Dataset construction (Monte Carlo return) -------
            # Every (prompt, completion) pair in this episode gets the SAME
            # final_profit as its reward signal.  Good episodes reinforce ALL
            # actions taken; bad episodes suppress them — Monte Carlo credit.
            mc_returns = [final_profit] * len(prompts)
            dataset = Dataset.from_dict({
                "prompt":     prompts,
                "completion": completions,
                "mc_return":  mc_returns,
            })

            # ---- PHASE 3: One epoch of GRPO training ----------------------
            trainer = GRPOTrainer(
                model=model,
                args=grpo_config,
                train_dataset=dataset,
                processing_class=tokenizer,
                reward_funcs=[reward_fn_mc_profit, reward_fn_tool_format, reward_fn_placement],
            )
            trainer.train()

            # ---- Record metrics -------------------------------------------
            all_profits.append(final_profit)
            # Average reward from trainer log
            ep_reward = 0.0
            for log in trainer.state.log_history:
                ep_reward = log.get("reward", log.get("rewards/reward_fn_mc_profit/mean", ep_reward))
            all_rewards.append(float(ep_reward))

            if ep % max(1, args.num_episodes // 10) == 0:
                print(
                    f"  Episode {ep:4d}/{args.num_episodes}  "
                    f"profit=${final_profit:>10,.0f}  "
                    f"steps={len(prompts)}  "
                    f"reward={float(ep_reward):+.4f}"
                )

    # Save model
    trainer.save_model(args.output_dir)
    print(f"\n[✓] Training complete. Model saved to {args.output_dir}")

    # -----------------------------------------------------------------------
    # Graphs — plot true episode profit over training iterations
    # -----------------------------------------------------------------------
    results = {
        "grpo_mc_training": {
            "rewards": all_rewards,
            "profits": all_profits,
        }
    }
    _plot_reward_curves(results, len(all_profits))
    _save_metrics(results)


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
