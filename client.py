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
        """Serialise a StaffingAction → CallToolAction dict sent over WebSocket."""
        return {
            "type":      "call_tool",
            "tool_name": action.tool,
            "arguments": action.params,
        }

    def _parse_result(self, payload: dict) -> StepResult[StaffingObservation]:
        """Deserialise server response → StepResult[StaffingObservation].

        Server sends CallToolObservation serialized as:
          {
            "observation": {
                "tool_name": ...,
                "result": {"_ctx": {step, cash, profit, cumulative_reward},
                           "tool_result": <tool-specific dict>},
                "error": null,
            },
            "reward": float,
            "done": bool,
          }
        """
        obs_dict    = payload.get("observation", {})
        
        # 1. Try to find nested result (from CallToolObservation)
        result_wrap = obs_dict.get("result") or obs_dict.get("metadata", {}).get("result")
        
        if isinstance(result_wrap, dict):
            ctx         = result_wrap.get("_ctx", {})
            tool_result = result_wrap.get("tool_result")
        else:
            ctx         = {}
            tool_result = result_wrap
            
        # 2. Fallback to top-level fields (from direct StaffingObservation or _ctx)
        # Priority: explicit ctx > top-level obs_dict
        step        = ctx.get("step")   if ctx.get("step") is not None   else obs_dict.get("step", 0)
        cash        = ctx.get("cash")   if ctx.get("cash") is not None   else obs_dict.get("cash", 0.0)
        profit      = ctx.get("profit") if ctx.get("profit") is not None else obs_dict.get("profit", 0.0)
        cum_reward  = ctx.get("cumulative_reward") if ctx.get("cumulative_reward") is not None else obs_dict.get("cumulative_reward", 0.0)
        
        # If tool_result is still none, maybe it's the obs_dict itself (if it's not a structured model)
        # or maybe there's a 'tool_result' field in obs_dict
        if tool_result is None:
            tool_result = obs_dict.get("tool_result")
            
        message     = obs_dict.get("message", "")
        reward      = payload.get("reward", 0.0)
        done        = payload.get("done", False)

        obs = StaffingObservation(
            step=int(step),
            cash=float(cash),
            profit=float(profit),
            cumulative_reward=float(cum_reward),
            tool_result=tool_result,
            message=str(message),
            reward=float(reward if reward is not None else 0.0),
            done=bool(done)
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

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
