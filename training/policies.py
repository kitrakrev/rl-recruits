"""
Heuristic baseline policies used by the dry-run simulator.

Three policies implement increasing levels of sophistication:
  random   → baseline noise floor
  greedy   → disciplined pipeline (interview → hire → place)
  optimal  → demand-aware, deadline-sorted, pass on unwinnable projects

All operate on a policy_state dict that caches the last known
candidate/project lists (to avoid spending episode steps on reads).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random as _random_mod
    from client import StaffingAgencyEnv


# Seniority ordering (used by compatibility checks)
_SEN_ORDER = {"junior": 0, "mid": 1, "senior": 2}

# Developer-type adjacency: which candidate types can fill which role types
_ADJACENT: dict[str, set[str]] = {
    "backend":     {"backend", "fullstack", "ml_engineer"},
    "frontend":    {"frontend", "fullstack"},
    "fullstack":   {"fullstack", "backend", "frontend"},
    "ml_engineer": {"ml_engineer", "backend"},
    "devops":      {"devops"},
}


def _role_is_open(r: dict) -> bool:
    if "is_filled" in r:
        return not r["is_filled"]
    return r.get("filled_count", 0) < r.get("headcount", 1)


def _can_fill(cand: dict, role: dict) -> bool:
    """Full compatibility check: developer type + skill score + seniority."""
    cand_type = cand.get("developer_type", "")
    role_type = role.get("developer_type", "")
    if role_type not in _ADJACENT.get(cand_type, {cand_type}):
        return False
    if cand.get("skill_score", 0) < role.get("min_skill_score", 0):
        return False
    cand_sen = _SEN_ORDER.get(cand.get("seniority_level", "junior"), 0)
    role_sen = _SEN_ORDER.get(role.get("seniority", "junior"), 0)
    return cand_sen >= role_sen


def _can_fill_type(cand_type: str, role_type: str) -> bool:
    return role_type in _ADJACENT.get(cand_type, {cand_type})


# ---------------------------------------------------------------------------
# Policy: random
# ---------------------------------------------------------------------------

def policy_random(env: "StaffingAgencyEnv", rng: "_random_mod.Random", state: dict):
    """Random baseline: hire anyone in pipeline, then randomly place or explore."""
    from models import StaffingAction

    candidates = state.get("candidates", [])
    projects   = state.get("projects", [])
    market     = state.get("market", [])

    for c in candidates:
        if c.get("status") == "in_pipeline":
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    hired = [c for c in candidates if c.get("status") == "hired"]
    if hired and projects:
        c = rng.choice(hired)
        for p in projects:
            for r in p.get("roles", []):
                if _role_is_open(r):
                    return StaffingAction(
                        tool="match_candidate_to_project",
                        params={"candidate_id": c["id"], "project_id": p["project_id"], "role_id": r["role_id"]},
                    )

    if market:
        c = rng.choice(market)
        return StaffingAction(tool="interview_candidate", params={"candidate_id": c["id"]})

    return StaffingAction(tool=rng.choice(["find_available_projects", "find_candidate"]), params={})


# ---------------------------------------------------------------------------
# Policy: greedy
# ---------------------------------------------------------------------------

def policy_greedy(env: "StaffingAgencyEnv", rng: "_random_mod.Random", state: dict):
    """
    Greedy: strict interview → hire → place pipeline.

    Checks type + skill + seniority compatibility before matching.
    Lets go of benched candidates with no viable open roles.
    """
    from models import StaffingAction

    candidates     = state.get("candidates", [])
    projects       = state.get("projects", [])
    market         = state.get("market", [])
    failed_matches = state.setdefault("_failed_matches", set())

    hired    = [c for c in candidates if c.get("status") == "hired"]
    pipeline = [c for c in candidates if c.get("status") == "in_pipeline"]
    all_avail = hired + pipeline + market

    open_roles = [
        (p, r) for p in projects for r in p.get("roles", []) if _role_is_open(r)
    ]

    step        = state.get("_step", 0)
    stale_off   = max(0, step - state.get("_last_proj_refresh", 0))

    # 0. Pass on unfillable / expiring projects
    for p in projects:
        open_p = [r for r in p.get("roles", []) if _role_is_open(r)]
        if not open_p:
            continue
        fillable = all(any(_can_fill(c, r) for c in all_avail) for r in open_p)
        dl_est   = max(0, p.get("deadline_remaining", 99) - stale_off)
        if not fillable or (dl_est <= 3 and not any(_can_fill(c, r) for c in hired for r in open_p)):
            return StaffingAction(tool="pass_on_project", params={"project_id": p["project_id"]})

    # 1. Hire anyone in pipeline
    for c in pipeline:
        return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # 2. Place compatible hired candidate
    for c in hired:
        for p, r in open_roles:
            key = (c["id"], r["role_id"])
            if _can_fill(c, r) and key not in failed_matches:
                return StaffingAction(
                    tool="match_candidate_to_project",
                    params={"candidate_id": c["id"], "project_id": p["project_id"], "role_id": r["role_id"]},
                )

    # 3. Release benched candidates with no viable role
    for c in hired:
        placeable = any(
            _can_fill(c, r) and (c["id"], r["role_id"]) not in failed_matches
            for _p, r in open_roles
        )
        if not placeable:
            return StaffingAction(tool="let_go_candidate", params={"candidate_id": c["id"]})

    # 4. Interview market candidates whose type matches an open role
    for mc in market:
        for _p, r in open_roles:
            if _can_fill(mc, r):
                return StaffingAction(tool="interview_candidate", params={"candidate_id": mc["id"]})

    # 5. Alternate cache refresh
    return _alternate_refresh(state)


# ---------------------------------------------------------------------------
# Policy: optimal heuristic
# ---------------------------------------------------------------------------

def policy_optimal(env: "StaffingAgencyEnv", rng: "_random_mod.Random", state: dict):
    """
    Demand-aware optimal: target easiest-to-seal projects first,
    only hire types that match open roles, pass expiring unwinnable projects.
    """
    from models import StaffingAction

    candidates     = state.get("candidates", [])
    projects       = state.get("projects", [])
    market         = state.get("market", [])
    failed_matches = state.setdefault("_failed_matches", set())

    step            = state.get("_step", 0)
    proj_ref_step   = state.get("_last_proj_refresh", 0)
    stale_off       = max(0, step - proj_ref_step)

    def est_deadline(p: dict) -> int:
        return max(0, p.get("deadline_remaining", 99) - stale_off)

    sorted_projects = sorted(projects, key=est_deadline, reverse=True)

    open_roles_all: list[tuple[dict, dict]] = []
    demand: dict[str, int] = {}
    for p in sorted_projects:
        for r in p.get("roles", []):
            if _role_is_open(r):
                dt = r.get("developer_type", "")
                demand[dt] = demand.get(dt, 0) + 1
                open_roles_all.append((p, r))

    hired    = [c for c in candidates if c.get("status") == "hired"]
    pipeline = [c for c in candidates if c.get("status") == "in_pipeline"]
    all_avail = hired + pipeline + market

    # 0. Pass on unfillable / expiring projects
    for p in sorted_projects:
        open_p = [r for r in p.get("roles", []) if _role_is_open(r)]
        if not open_p:
            continue
        fillable  = all(any(_can_fill(c, r) for c in all_avail) for r in open_p)
        dl_est    = est_deadline(p)
        if not fillable or (dl_est <= 3 and not any(_can_fill(c, r) for c in hired for r in open_p)):
            return StaffingAction(tool="pass_on_project", params={"project_id": p["project_id"]})

    # 1. Place any hired candidate into the best matching role
    for p, r in open_roles_all:
        for c in hired:
            key = (c["id"], r["role_id"])
            if _can_fill(c, r) and key not in failed_matches:
                return StaffingAction(
                    tool="match_candidate_to_project",
                    params={"candidate_id": c["id"], "project_id": p["project_id"], "role_id": r["role_id"]},
                )

    # 2. Hire pipeline candidates whose type matches demand
    for c in pipeline:
        if any(_can_fill_type(c.get("developer_type", ""), dt) for dt in demand):
            return StaffingAction(tool="hire_candidate", params={"candidate_id": c["id"]})

    # 3. Release non-placeable benched candidates
    for c in hired:
        placeable = any(
            _can_fill(c, r) and (c["id"], r["role_id"]) not in failed_matches
            for _p, r in open_roles_all
        )
        if not placeable:
            return StaffingAction(tool="let_go_candidate", params={"candidate_id": c["id"]})

    # 4. Interview highest-skill market candidates matching demand
    market_sorted = sorted(market, key=lambda c: c.get("skill_score", 0), reverse=True)
    for mc in market_sorted:
        for _p, r in open_roles_all:
            if _can_fill(mc, r):
                return StaffingAction(tool="interview_candidate", params={"candidate_id": mc["id"]})

    # 5. Alternate cache refresh
    return _alternate_refresh(state)


# ---------------------------------------------------------------------------
# Helper: alternate between project and market refresh to avoid repeat penalty
# ---------------------------------------------------------------------------

def _alternate_refresh(state: dict):
    from models import StaffingAction

    last = state.get("_last_refresh_tool", "")
    step     = state.get("_step", 0)
    proj_age = step - state.get("_last_proj_refresh", -99)
    mkt_age  = step - state.get("_last_mkt_refresh", -99)

    if last == "find_available_projects":
        state["_last_refresh_tool"] = "find_candidate"
        return StaffingAction(tool="find_candidate", params={})
    if last == "find_candidate":
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    if proj_age >= mkt_age:
        state["_last_refresh_tool"] = "find_available_projects"
        return StaffingAction(tool="find_available_projects", params={})
    state["_last_refresh_tool"] = "find_candidate"
    return StaffingAction(tool="find_candidate", params={})
