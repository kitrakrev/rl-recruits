"""
Typed Action, Observation, and State for the Staffing Agency OpenEnv.

These are the public API types used by:
  - client.py  (agent sends Action, receives Observation/State)
  - server/app.py  (registered with create_app)
  - training/train_grpo.py  (reward functions receive Observation)

Following openenv.md spec exactly.
"""
from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action — what the agent sends
# ---------------------------------------------------------------------------

class StaffingAction(Action):
    """
    A single tool call issued by the staffing agent.

    The agent calls one of 17 tools per action:
      GET  tools (no world tick): get_agency_state, get_client_state, ...
      EXEC tools (advances time):  find_candidate, hire_candidate, ...

    Example:
        StaffingAction(tool="find_candidate", params={"developer_type": "backend"})
        StaffingAction(tool="match_candidate_to_project", params={
            "candidate_id": "C-BA-abc12345",
            "project_id":   "P-CL-FIN-01-xyz",
            "role_id":      "R-P-CL-FIN-01-xyz-0"
        })
    """
    tool: str
    params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Observation — what the agent receives back
# ---------------------------------------------------------------------------

class StaffingObservation(Observation):
    """
    Observation returned after every step.

    Always includes reward (scaled P&L delta) and done (episode over?).
    tool_result holds the structured dict returned by the called tool.
    """
    # Inherited from Observation: done: bool, reward: float | None
    step: int = 0
    cash: float = 0.0
    profit: float = 0.0
    cumulative_reward: float = 0.0
    tool_result: Optional[Any] = None     # structured dict from the tool call
    message: str = ""                     # human-readable context (on reset)


# ---------------------------------------------------------------------------
# State — internal episode tracking (never sent to agent directly)
# ---------------------------------------------------------------------------

class StaffingState(State):
    """
    Full internal episode state — returned from GET /state.
    Visible to the training loop, not to the agent during inference.
    """
    # Inherited from State: episode_id: str | None, step_count: int
    cash: float = 50_000.0
    revenue: float = 0.0
    costs: float = 0.0
    num_placed: int = 0
    num_hired: int = 0
    num_benched: int = 0
    avg_satisfaction: float = 0.75
    cumulative_reward: float = 0.0
    done: bool = False
