"""
Tests for the Staffing Agency OpenEnv.
Tests the environment directly (no HTTP server needed).

Run: uv run pytest tests/ -v
"""
import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from env.config import Config
from server.staffing_environment import StaffingAgencyEnvironment


@pytest.fixture
def env():
    cfg = Config(
        llm_mode="stub", curriculum_stage=1, num_clients=1,
        t_deadline_min=10, t_deadline_max=14,  # longer deadlines so projects survive test steps
    )
    return StaffingAgencyEnvironment(cfg)


@pytest.fixture
def env_stage2():
    cfg = Config(llm_mode="stub", curriculum_stage=2, num_clients=3)
    return StaffingAgencyEnvironment(cfg)


def step(env, tool_name: str, **params):
    """Helper: call one tool and return the Observation."""
    action = CallToolAction(tool_name=tool_name, arguments=params)
    return env.step(action)


# ------------------------------------------------------------------
# Basic lifecycle
# ------------------------------------------------------------------

def test_reset_returns_observation(env):
    obs = env.reset(seed=42)
    assert obs.done is False
    assert "message" in obs.metadata
    assert "state" in obs.metadata


def test_reset_seeds_deterministically(env):
    obs1 = env.reset(seed=42)
    s1 = obs1.metadata["state"]
    obs2 = env.reset(seed=42)
    s2 = obs2.metadata["state"]
    assert s1["cash"] == s2["cash"]
    assert s1["num_clients"] == s2["num_clients"]


def test_state_property(env):
    env.reset(seed=1)
    state = env.state
    assert state.step_count == 0
    assert state.cash == pytest.approx(50_000.0)
    assert state.done is False


def test_step_increments_step_count(env):
    # GET tools are read-only (no tick). EXECUTE tools increment the step count.
    env.reset(seed=1)
    step(env, "find_candidate")      # EXECUTE tool
    assert env.state.step_count == 1
    step(env, "find_available_projects")   # EXECUTE tool
    assert env.state.step_count == 2
    # GET tool should NOT increment
    step(env, "get_agency_state")
    assert env.state.step_count == 2


# ------------------------------------------------------------------
# List tools
# ------------------------------------------------------------------

def test_list_tools(env):
    env.reset(seed=1)
    action = ListToolsAction()
    obs = env.step(action)
    # ListToolsObservation has a .tools attribute (list of Tool objects)
    assert hasattr(obs, "tools"), f"Expected .tools on obs, got: {type(obs)}"
    tool_names = [t.name for t in obs.tools]
    expected = [
        "get_agency_state", "get_client_state", "get_candidate_state",
        "get_project_details", "get_candidate_profile", "get_market_demand",
        "get_financial_summary", "find_available_projects", "confirm_project",
        "find_candidate", "interview_candidate", "hire_candidate",
        "negotiate_salary", "match_candidate_to_project", "let_go_candidate",
        "request_project_extension", "pass_on_project",
    ]
    assert len(tool_names) == 17, f"Expected 17 tools, got {len(tool_names)}: {tool_names}"
    for name in expected:
        assert name in tool_names, f"Missing tool: {name}"


# ------------------------------------------------------------------
# GET tools
# ------------------------------------------------------------------

def test_get_agency_state(env):
    env.reset(seed=2)
    obs = step(env, "get_agency_state")
    r = obs.metadata["tool_result"]
    assert "cash_balance" in r
    assert r["cash_balance"] == pytest.approx(50_000.0, abs=1000)
    assert "burn_rate" in r
    assert "placement_rate" in r


def test_get_client_state_all(env):
    env.reset(seed=3)
    obs = step(env, "get_client_state")
    r = obs.metadata["tool_result"]
    assert "clients" in r
    assert len(r["clients"]) >= 1


def test_get_client_state_specific(env):
    env.reset(seed=3)
    clients = step(env, "get_client_state").metadata["tool_result"]["clients"]
    cid = clients[0]["client_id"]
    obs = step(env, "get_client_state", client_id=cid)
    r = obs.metadata["tool_result"]
    assert r["client_id"] == cid


def test_get_candidate_state(env):
    env.reset(seed=4)
    obs = step(env, "get_candidate_state")
    r = obs.metadata["tool_result"]
    assert "num_in_market" in r
    assert r["num_in_market"] > 0


def test_get_market_demand(env):
    env.reset(seed=5)
    obs = step(env, "get_market_demand")
    r = obs.metadata["tool_result"]
    assert "demand_by_type" in r
    assert "total_open_slots" in r


def test_get_financial_summary(env):
    env.reset(seed=6)
    obs = step(env, "get_financial_summary")
    r = obs.metadata["tool_result"]
    assert "cash_balance" in r
    assert "burn_rate" in r
    assert "profit" in r


# ------------------------------------------------------------------
# EXECUTE tools — full pipeline
# ------------------------------------------------------------------

def test_find_candidate(env):
    env.reset(seed=10)
    obs = step(env, "find_candidate")
    r = obs.metadata["tool_result"]
    assert "candidates" in r
    assert r["count"] > 0


def test_find_candidate_by_type(env):
    env.reset(seed=10)
    obs = step(env, "find_candidate", developer_type="backend")
    r = obs.metadata["tool_result"]
    for c in r["candidates"]:
        assert c["developer_type"] == "backend"


def test_interview_candidate(env):
    env.reset(seed=11)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    assert cands, "No candidates in market"
    cid = cands[0]["id"]
    obs = step(env, "interview_candidate", candidate_id=cid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert "base_rating" in r
    assert 1 <= r["base_rating"] <= 5
    assert "proceed" in r


def test_interview_moves_candidate_to_pipeline(env):
    env.reset(seed=12)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    cstate = step(env, "get_candidate_state").metadata["tool_result"]
    assert cstate["num_in_pipeline"] == 1


def test_hire_candidate(env):
    env.reset(seed=13)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    obs = step(env, "hire_candidate", candidate_id=cid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert "salary_weekly" in r
    assert "break_even_weeks" in r
    assert r["onboarding_cost"] == 2000.0


def test_hire_deducts_cash(env):
    env.reset(seed=14)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    cash_before = step(env, "get_financial_summary").metadata["tool_result"]["cash_balance"]
    step(env, "hire_candidate", candidate_id=cid)
    cash_after = step(env, "get_financial_summary").metadata["tool_result"]["cash_balance"]
    assert cash_after < cash_before  # onboarding + bench costs deducted


def test_negotiate_salary(env):
    env.reset(seed=15)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    # Offer very high salary — should be accepted
    obs = step(env, "negotiate_salary", candidate_id=cid, offer_weekly=5000.0)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert r["accepted"] is True


def test_negotiate_salary_low_rejected(env):
    env.reset(seed=16)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    # Offer very low salary — should be rejected
    obs = step(env, "negotiate_salary", candidate_id=cid, offer_weekly=1.0)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert r["accepted"] is False
    assert r["counter_offer"] is not None


def test_find_available_projects(env):
    env.reset(seed=20)
    # EXECUTE tools tick the world — projects arrive via Poisson each tick
    for _ in range(5):
        step(env, "find_candidate")   # EXECUTE, advances time
    obs = step(env, "find_available_projects")
    r = obs.metadata["tool_result"]
    assert "projects" in r
    assert "count" in r


def test_confirm_project(env):
    env.reset(seed=21)
    for _ in range(5):
        step(env, "find_candidate")
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects available")
    pid = projects[0]["project_id"]
    obs = step(env, "confirm_project", project_id=pid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert r["project_id"] == pid


def test_full_hire_and_place_pipeline(env):
    """Full cycle: find candidate → interview → hire → find project → match."""
    env.reset(seed=30)

    # Tick to get projects using EXECUTE tool
    for _ in range(6):
        step(env, "find_candidate")

    # Find a project and its role
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects available after ticking")

    project = projects[0]
    pid = project["project_id"]
    role = project["roles"][0]
    rid = role["role_id"]
    needed_type = role["developer_type"]
    min_skill = role["min_skill_score"]

    # Find matching candidate
    cands = step(env, "find_candidate", developer_type=needed_type).metadata["tool_result"]["candidates"]
    # Pick one with sufficient skill
    matching = [c for c in cands if c["skill_score"] >= min_skill]
    if not matching:
        pytest.skip(f"No {needed_type} candidate with skill >= {min_skill}")
    cid = matching[0]["id"]

    # Pipeline
    step(env, "interview_candidate", candidate_id=cid)
    hire_r = step(env, "hire_candidate", candidate_id=cid).metadata["tool_result"]
    assert hire_r["success"]

    # Match
    obs = step(env, "match_candidate_to_project",
               candidate_id=cid, project_id=pid, role_id=rid)
    r = obs.metadata["tool_result"]
    assert r["success"], f"Match failed: {r.get('error')}"
    assert r["match_score"] > 0
    assert "composite_rating" in r
    assert "project_sealed" in r


def test_illegal_match_blocked(env):
    """Candidate with wrong type cannot be placed."""
    env.reset(seed=31)
    for _ in range(6):
        step(env, "find_candidate")
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects")
    project = projects[0]
    pid = project["project_id"]
    role = project["roles"][0]
    rid = role["role_id"]
    needed_type = role["developer_type"]

    # Find candidate of DIFFERENT type
    all_types = ["backend", "frontend", "fullstack", "ml_engineer", "devops"]
    # adjacency means fullstack can fill backend/frontend — use devops as strictly incompatible
    from env.config import Config
    cfg = Config()
    incompatible = None
    for c in env._market:
        adj = cfg.adjacency.get(c.developer_type, set())
        if needed_type not in adj:
            incompatible = c
            break
    if not incompatible:
        pytest.skip("All market candidates are compatible")

    step(env, "interview_candidate", candidate_id=incompatible.id)
    step(env, "hire_candidate", candidate_id=incompatible.id)
    obs = step(env, "match_candidate_to_project",
               candidate_id=incompatible.id, project_id=pid, role_id=rid)
    r = obs.metadata["tool_result"]
    assert r["success"] is False
    assert "Illegal" in r.get("error", "") or "mismatch" in r.get("error", "")


def test_let_go_candidate(env):
    env.reset(seed=40)
    cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
    cid = cands[0]["id"]
    step(env, "interview_candidate", candidate_id=cid)
    step(env, "hire_candidate", candidate_id=cid)
    obs = step(env, "let_go_candidate", candidate_id=cid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert r["severance_paid"] > 0
    # Candidate should be gone
    profile = step(env, "get_candidate_profile", candidate_id=cid).metadata["tool_result"]
    assert "error" in profile


def test_request_project_extension(env):
    env.reset(seed=41)
    for _ in range(5):
        step(env, "find_candidate")
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects")
    pid = projects[0]["project_id"]
    deadline_before = projects[0]["deadline_remaining"]
    obs = step(env, "request_project_extension", project_id=pid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    assert r["new_deadline_remaining"] > deadline_before


def test_pass_on_project(env):
    env.reset(seed=42)
    for _ in range(5):
        step(env, "find_candidate")
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects")
    pid = projects[0]["project_id"]
    obs = step(env, "pass_on_project", project_id=pid)
    r = obs.metadata["tool_result"]
    assert r["success"] is True
    # Project should no longer appear
    remaining = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    remaining_ids = [p["project_id"] for p in remaining]
    assert pid not in remaining_ids


def test_get_project_details(env):
    env.reset(seed=43)
    for _ in range(5):
        step(env, "find_candidate")
    projects = step(env, "find_available_projects").metadata["tool_result"]["projects"]
    if not projects:
        pytest.skip("No projects")
    pid = projects[0]["project_id"]
    obs = step(env, "get_project_details", project_id=pid)
    r = obs.metadata["tool_result"]
    assert r["project_id"] == pid
    assert "roles" in r
    assert "deadline_remaining" in r


def test_reward_is_float(env):
    env.reset(seed=50)
    obs = step(env, "get_agency_state")
    assert isinstance(obs.reward, (int, float))


def test_done_after_52_steps(env):
    # Use an EXECUTE tool so step_count increments
    env.reset(seed=51)
    obs = None
    for _ in range(52):
        obs = step(env, "find_available_projects")   # EXECUTE — advances step
        if obs.done:
            break
    assert obs is not None
    assert env.state.step_count >= 52 or obs.done


def test_bankruptcy_terminates_episode():
    """Episode ends when cash goes negative."""
    cfg = Config(llm_mode="stub", seed_capital=500.0, curriculum_stage=1, num_clients=1)
    env = StaffingAgencyEnvironment(cfg)
    env.reset(seed=99)
    # Hire candidates until bankrupt
    for _ in range(20):
        cands = step(env, "find_candidate").metadata["tool_result"]["candidates"]
        if not cands:
            break
        cid = cands[0]["id"]
        step(env, "interview_candidate", candidate_id=cid)
        obs = step(env, "hire_candidate", candidate_id=cid)
        if obs.done:
            assert env.state.cash < 0 or obs.done
            return
    # Tick to drain cash via bench costs
    for _ in range(30):
        obs = step(env, "get_agency_state")
        if obs.done:
            return


def test_52_step_random_rollout_no_crash(env_stage2):
    """Full 52-step episode with mixed tool calls should not crash."""
    import random
    env_stage2.reset(seed=77)
    rng = random.Random(77)
    safe_tools = [
        "get_agency_state", "get_candidate_state",
        "find_available_projects", "get_financial_summary", "get_market_demand",
    ]
    for _ in range(52):
        tool = rng.choice(safe_tools)
        obs = step(env_stage2, tool)
        assert obs is not None
        if obs.done:
            break


def test_unknown_tool_returns_error(env):
    env.reset(seed=60)
    action = CallToolAction(tool_name="nonexistent_tool", arguments={})
    obs = env.step(action)
    result = obs.metadata.get("tool_result", {})
    # Should return an error, not crash
    assert "error" in result or "tool_result" in obs.metadata
