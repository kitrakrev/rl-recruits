"""
StaffingAgencyEnvironment — OpenEnv MCPEnvironment for the Staffing Agency RL task.

Implements the OpenEnv contract:
  - Inherits MCPEnvironment
  - Tools registered via @mcp.tool (FastMCP)
  - reset() / step() / state property
  - Proper Observation with reward + done

Theme: Multi-Agent Interactions + Long-Horizon Planning (Scale AI / Mercor sub-theme)
Agent must manage a multi-actor system (candidates + clients) over a 52-step horizon.
"""
from __future__ import annotations

import os
from uuid import uuid4
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from env.config import Config
from env.llm import LLMRouter
from models import StaffingState, StaffingObservation


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class StaffingAgencyEnvironment(MCPEnvironment):
    """
    A multi-actor, long-horizon staffing agency environment.

    The LLM agent acts as a staffing agency CEO:
    - Interacts with CLIENTS (demand side) who submit multi-role projects
    - Manages CANDIDATES (supply side) through interview → hire → placement pipeline
    - Must balance bench costs vs. revenue to maximise profit over 52 weeks

    Innovation: Sparse multi-role project sealing creates long-horizon credit
    assignment challenges. The agent must reason about future demand, candidate
    patience decay, and client satisfaction simultaneously.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    # GET-only tools that produce zero world progress
    _PASSIVE_TOOLS = frozenset({
        "get_agency_state", "get_client_state", "get_candidate_state",
        "get_project_details", "get_candidate_profile",
        "get_market_demand", "get_financial_summary",
    })
    _PASSIVE_STREAK_PENALTY = -50.0   # $ per consecutive passive-only turn above threshold
    _PASSIVE_STREAK_THRESHOLD = 3     # free passive turns before penalty kicks in
    _REPEAT_CALL_PENALTY = -100.0     # penalty for calling the exact same tool twice in a row

    def __init__(self, config: Config | None = None):
        self._config = config or Config(
            llm_mode=os.getenv("LLM_MODE", "stub"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        self._llm = LLMRouter(self._config)
        from env.core import StaffingCore
        self.core = StaffingCore(self._config, self._llm, env_type="mcp")
        self._state = StaffingState()
        self._passive_streak: int = 0   # consecutive GET-only turns this episode
        self._last_tool: str = ""        # last tool called, for repeat detection

        import random
        self._rng = random.Random()

        mcp = FastMCP("staffing_agency")
        self._register_tools(mcp)
        super().__init__(mcp)

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> StaffingObservation:
        self.core.reset(seed)
        self._passive_streak = 0
        self._last_tool = ""
        episode_id = episode_id or str(uuid4())
        self._state = StaffingState(
            episode_id=episode_id,
            step_count=0,
            cash=self.core.cash,
            revenue=0.0,
            costs=0.0,
            done=False,
        )
        snap = self._snapshot()
        
        return StaffingObservation(
            done=False,
            reward=0.0,
            step=snap["step"],
            cash=snap["cash"],
            profit=snap["profit"],
            cumulative_reward=0.0,
            tool_result={"message": "Environment reset. Current week: 1"},
            message="Welcome to Staffing Agency RL. Environment reset."
        )

    # GET tools are read-only — no world tick, no step increment
    # Includes admin override tools (UI-only) so they don't burn episode steps
    _GET_TOOLS = frozenset({
        "get_agency_state", "get_client_state", "get_candidate_state",
        "get_project_details", "get_candidate_profile",
        "get_market_demand", "get_financial_summary",
        "_override_cash", "_override_satisfaction",
    })

    def step(self, action: Any, timeout_s: float | None = None, **kwargs) -> Observation:
        """Route MCP tool calls + run world tick.

        ListToolsAction → pass through raw ListToolsObservation (no tick).
        GET CallToolAction → run tool only, no tick (pure observation).
        EXECUTE CallToolAction → run tool + world tick + step increment.
        """
        # Tool discovery: no tick
        if isinstance(action, ListToolsAction):
            return super().step(action, timeout_s=timeout_s, **kwargs)

        # EXECUTE CallToolAction → No longer automatically ticks world.
        # Step increment and world_tick are now handled by 'advance_week' tool.
        pass

        # Let MCPEnvironment route EXECUTE CallToolAction
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Sync state with core after action (tool might have changed cash/etc)
        self._state.step_count = self.core.step_count
        self._state.cash = self.core.cash
        self._state.revenue = self.core.revenue
        self._state.costs = self.core.costs
        
        # Repeat-call penalty: penalize calling the same tool twice in a row
        tool_name = action.tool_name if hasattr(action, "tool_name") else ""
        repeat_penalty = 0.0
        if tool_name and tool_name == self._last_tool:
            repeat_penalty = self._REPEAT_CALL_PENALTY
            self.core.costs += abs(repeat_penalty)
        self._last_tool = tool_name

        # Passive-streak penalty: discourage consecutive GET-only turns
        if tool_name in self._PASSIVE_TOOLS:
            self._passive_streak += 1
        else:
            self._passive_streak = 0  # reset on any active action

        passive_penalty = 0.0
        if self._passive_streak > self._PASSIVE_STREAK_THRESHOLD:
            passive_penalty = self._PASSIVE_STREAK_PENALTY
            self.core.costs += abs(passive_penalty)  # reflect in financials

        # Re-sync costs after penalties (penalties mutate core.costs above)
        self._state.costs = self.core.costs

        # Reward is now primarily tool-based OR tick-based (if tool was advance_week)
        total_reward = (obs.reward or 0.0) + passive_penalty + repeat_penalty
        self._state.cumulative_reward += total_reward
        
        done = (
            self._state.cash < 0
            or self._state.step_count >= self._config.episode_steps
        )
        self._state.done = done
        self._state.num_hired = sum(1 for c in self.core.candidates.values() if c.status in ("hired", "placed"))
        self._state.num_placed = sum(1 for c in self.core.candidates.values() if c.status == "placed")
        self._state.num_benched = sum(1 for c in self.core.candidates.values() if c.status == "hired")
        active = [cl for cl in self.core.clients if not cl.churn_risk]
        self._state.avg_satisfaction = sum(cl.satisfaction_score for cl in active) / len(active) if active else 0.0

        # Unwrap CallToolObservation → plain dict
        tool_result = None
        if hasattr(obs, "result") and obs.result is not None:
            r = obs.result
            if hasattr(r, "data"):
                tool_result = r.data
            elif hasattr(r, "structured_content"):
                tool_result = r.structured_content
            else:
                tool_result = r
        elif hasattr(obs, "metadata"):
            tool_result = obs.metadata

        return CallToolObservation(
            tool_name=action.tool_name,
            result={
                "_ctx": {
                    "step": self._state.step_count,
                    "cash": round(self._state.cash, 2),
                    "profit": round(self._state.revenue - self._state.costs, 2),
                    "cumulative_reward": round(self._state.cumulative_reward, 4),
                },
                "tool_result": tool_result,
            },
            done=done,
            reward=total_reward,
        )

    @property
    def state(self) -> StaffingState:
        return self._state

    def _step_impl(self, action: Any, timeout_s: float | None = None, **kwargs) -> Observation:
        """Fallback for unknown action types."""
        return Observation(
            done=False, reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}"},
        )

    # ------------------------------------------------------------------
    # Tool registration via FastMCP
    # ------------------------------------------------------------------

    def _register_tools(self, mcp: FastMCP) -> None:
        env = self

        @mcp.tool()
        def get_agency_state() -> dict: return env.core.tool_get_agency_state()["state"]

        @mcp.tool()
        def get_client_state(client_id: str = "") -> dict: 
            res = env.core.tool_get_client_state(client_id)
            if "error" in res: return res
            if isinstance(res.get("state"), list):
                return {"clients": res["state"]}
            return res.get("state") or res

        @mcp.tool()
        def get_candidate_state() -> dict: 
            res = env.core.tool_get_candidate_state()
            res.pop("reward", None)
            res.pop("success", None)
            res.pop("state", None)
            return res

        @mcp.tool()
        def get_project_details(project_id: str) -> dict:
            res = env.core.tool_get_project_details(project_id)
            if "error" in res: return {"error": res["error"]}
            return dict(res["project"])

        @mcp.tool()
        def get_candidate_profile(candidate_id: str) -> dict:
            res = env.core.tool_get_candidate_profile(candidate_id)
            if "error" in res: return {"error": res["error"]}
            return dict(res["candidate"])

        @mcp.tool()
        def get_market_demand() -> dict: return {"demand_by_type": env.core.tool_get_market_demand()["demand_by_type"], "total_open_slots": env.core.tool_get_market_demand()["total_open_slots"]}

        @mcp.tool()
        def get_financial_summary() -> dict: return env.core.tool_get_financial_summary()["summary"]

        @mcp.tool()
        def find_available_projects() -> dict:
            r = env.core.tool_find_available_projects()
            return {"projects": r["projects"], "count": r["count"]}

        @mcp.tool()
        def confirm_project(project_id: str) -> dict: 
            res = env.core.tool_confirm_project(project_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def find_candidate(developer_type: str = "") -> dict: 
            r = env.core.tool_find_candidate(developer_type)
            return {"candidates": r["candidates"], "count": r["count"]}

        @mcp.tool()
        def interview_candidate(candidate_id: str) -> dict: 
            res = env.core.tool_interview_candidate(candidate_id)
            if "reward" in res: res.pop("reward")
            if "result" in res: res.pop("result")
            return res

        @mcp.tool()
        def hire_candidate(candidate_id: str) -> dict: 
            res = env.core.tool_hire_candidate(candidate_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def negotiate_salary(candidate_id: str, offer_weekly: float) -> dict: 
            res = env.core.tool_negotiate_salary(candidate_id, offer_weekly)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def match_candidate_to_project(candidate_id: str, project_id: str, role_id: str) -> dict: 
            res = env.core.tool_match_candidate_to_project(candidate_id, project_id, role_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def let_go_candidate(candidate_id: str) -> dict: 
            res = env.core.tool_let_go_candidate(candidate_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def request_project_extension(project_id: str) -> dict: 
            res = env.core.tool_request_project_extension(project_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def pass_on_project(project_id: str) -> dict: 
            res = env.core.tool_pass_on_project(project_id)
            if "reward" in res: res.pop("reward")
            return res

        @mcp.tool()
        def advance_week() -> dict:
            """Advance simulation to the next week (ticks world, pays costs/revenue)."""
            return env.core.tool_advance_week()

        @mcp.tool()
        def _override_cash(amount: float) -> dict:
            old = env.core.cash
            env.core.cash = float(amount)
            env._state.cash = env.core.cash
            return {"success": True, "old_cash": round(old, 2), "new_cash": round(env.core.cash, 2)}

        @mcp.tool()
        def _override_satisfaction(client_id: str, score: float) -> dict:
            client = next((c for c in env.core.clients if c.client_id == client_id), None)
            if not client: return {"success": False, "error": f"Client {client_id} not found"}
            old = client.satisfaction_score
            client.satisfaction_score = max(0.0, min(1.0, float(score)))
            client.churn_risk = client.satisfaction_score < env._config.churn_threshold
            return {"success": True, "client_id": client_id, "old_score": round(old, 3), "new_score": round(client.satisfaction_score, 3), "churn_risk": client.churn_risk}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        """Compact state snapshot for observation metadata."""
        return {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "cash": round(self._state.cash, 2),
            "profit": round(self._state.revenue - self._state.costs, 2),
            "num_clients": len(self.core.clients),
            "num_candidates": len(self.core.candidates),
            "done": self._state.done,
        }
