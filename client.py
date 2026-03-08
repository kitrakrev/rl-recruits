"""
StaffingAgencyEnv — OpenEnv client for the Staffing Agency environment.

Follows the EnvClient pattern from openenv.md exactly:
  - Inherits EnvClient[StaffingAction, StaffingObservation, StaffingState]
  - Implements _step_payload, _parse_result, _parse_state
  - Typed Action / Observation / State from models.py

Usage (sync — recommended for training):
    from client import StaffingAgencyEnv
    from models import StaffingAction

    with StaffingAgencyEnv(base_url="http://localhost:8000") as env:
        result = env.reset(seed=42)
        print(result.observation.message)

        result = env.step(StaffingAction(tool="get_agency_state"))
        print(result.observation.tool_result)

        result = env.step(StaffingAction(
            tool="find_candidate",
            params={"developer_type": "backend"}
        ))
        print(result.observation.tool_result["candidates"])

        result = env.step(StaffingAction(
            tool="match_candidate_to_project",
            params={
                "candidate_id": "C-BA-abc12345",
                "project_id":   "P-CL-FIN-01-xyz",
                "role_id":      "R-P-CL-FIN-01-xyz-0",
            }
        ))
        print(f"reward={result.reward}  done={result.done}")

Usage (async):
    import asyncio
    from client import StaffingAgencyEnv
    from models import StaffingAction

    async def main():
        async with StaffingAgencyEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=42)
            result = await env.step(StaffingAction(tool="get_agency_state"))
            state  = await env.state()
            print(f"step={state.step_count}  cash=${state.cash:,.0f}")

    asyncio.run(main())

Connect from HuggingFace Hub (once deployed):
    env = await StaffingAgencyEnv.from_env("openenv/staffing-agency")
"""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import StaffingAction, StaffingObservation, StaffingState


class StaffingAgencyEnv(EnvClient[StaffingAction, StaffingObservation, StaffingState]):
    """
    Client for the Staffing Agency RL environment.

    Connects via WebSocket to the OpenEnv server (server/app.py).
    The agent issues StaffingAction(tool=..., params=...) and receives
    StaffingObservation with reward, done, cash, profit, and tool_result.
    """

    def _step_payload(self, action: StaffingAction) -> dict:
        """Serialise a StaffingAction → JSON dict sent over WebSocket."""
        return {
            "tool":   action.tool,
            "params": action.params,
        }

    def _parse_result(self, payload: dict) -> StepResult[StaffingObservation]:
        """Deserialise server response → StepResult[StaffingObservation]."""
        obs = StaffingObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            step=payload.get("step", 0),
            cash=payload.get("cash", 0.0),
            profit=payload.get("profit", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            tool_result=payload.get("tool_result"),
            message=payload.get("message", ""),
        )
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict) -> StaffingState:
        """Deserialise /state response → StaffingState."""
        return StaffingState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            cash=payload.get("cash", 0.0),
            revenue=payload.get("revenue", 0.0),
            costs=payload.get("costs", 0.0),
            num_placed=payload.get("num_placed", 0),
            num_hired=payload.get("num_hired", 0),
            num_benched=payload.get("num_benched", 0),
            avg_satisfaction=payload.get("avg_satisfaction", 0.75),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
        )
