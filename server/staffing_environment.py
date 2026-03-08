"""
StaffingAgencyEnvironment — OpenEnv MCPEnvironment for the Staffing Agency RL task.

Implements the OpenEnv contract:
  - Inherits MCPEnvironment
  - Tools registered via @mcp.tool (FastMCP)
  - reset() / step() / state property
  - Proper Observation with reward + done

Theme: Multi-Agent Interactions + Long-Horizon Planning (Scale AI / Mercor sub-theme)
Agent must manage a multi-actor system (candidates + clients) over a 52-step horizon.

Reward flow (IMPORTANT):
  MCPEnvironment.step() executes the MCP tool and returns a CallToolObservation
  whose .reward is always 0 — the framework does NOT extract reward from the tool
  result dict. Instead, each registered tool wrapper stores the reward it computed
  in env._last_tool_reward immediately before stripping it from the dict sent to
  the agent. staffing_environment.step() then reads _last_tool_reward to build the
  true total_reward. This keeps reward signal out of the agent's conversation while
  still propagating it correctly to the training loop.
"""
from __future__ import annotations

import logging
import os
import traceback
from uuid import uuid4
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from env.config import Config
from env.llm import LLMRouter
from models import StaffingState, StaffingObservation

logger = logging.getLogger("staffing.env")


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
        "get_market_demand", "get_financial_summary", "get_candidate_types",
        "find_available_projects",
    })

    # Canonical valid arguments per tool. Any extra args the LLM hallucinates
    # are silently stripped before FastMCP validates the call — preventing
    # ValidationError and the zero-reward black hole it creates.
    _TOOL_VALID_ARGS: dict[str, frozenset[str]] = {
        "get_agency_state":          frozenset(),
        "get_client_state":          frozenset({"client_id"}),
        "get_candidate_state":       frozenset(),
        "get_financial_summary":     frozenset(),
        "get_market_demand":         frozenset(),
        "get_candidate_types":       frozenset(),
        "find_candidate":            frozenset({"developer_type", "seniority_level", "min_skill_score", "min_composite_rating"}),
        "interview_candidate":       frozenset({"candidate_id"}),
        "hire_candidate":            frozenset({"candidate_id"}),
        "negotiate_salary":          frozenset({"candidate_id", "offer_weekly"}),
        "match_candidate_to_project":frozenset({"candidate_id", "project_id", "role_id"}),
        "let_go_candidate":          frozenset({"candidate_id"}),
        "confirm_project":           frozenset({"project_id"}),
        "pass_on_project":           frozenset({"project_id"}),
        "request_project_extension": frozenset({"project_id"}),
        "find_available_projects":   frozenset(),
        "advance_week":              frozenset(),
    }

    def __init__(self, config: Config | None = None):
        self._config = config or Config(
            llm_mode=os.getenv("LLM_MODE", "stub"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        logger.info(
            "[env.__init__] Creating environment  llm_mode=%s  stage=%s  capital=$%s  steps=%s",
            self._config.llm_mode, self._config.curriculum_stage,
            f"{self._config.seed_capital:,.0f}", self._config.episode_steps,
        )
        try:
            self._llm = LLMRouter(self._config)
            from env.core import StaffingCore
            self.core = StaffingCore(self._config, self._llm, env_type="mcp")
        except Exception:
            logger.exception("[env.__init__] FAILED to init StaffingCore/LLMRouter")
            raise

        self._state = StaffingState()
        self._passive_streak: int = 0   # consecutive GET-only turns this episode
        self._last_tool: str = ""        # last tool called, for repeat detection
        self._last_tool_failed: bool = False  # was the last call a failure?

        # Reward capture: each MCP tool wrapper stores its reward here before
        # stripping the "reward" key from the dict it returns to the agent.
        # step() reads this after super().step() executes the tool.
        self._last_tool_reward: float = 0.0

        import random
        self._rng = random.Random()

        try:
            mcp = FastMCP("staffing_agency")
            self._register_tools(mcp)
            super().__init__(mcp)
            logger.info("[env.__init__] Tools registered — environment ready")
        except Exception:
            logger.exception("[env.__init__] FAILED during tool registration or super().__init__")
            raise

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> StaffingObservation:
        logger.info("[env.reset] seed=%s  episode_id=%s", seed, episode_id)
        try:
            self.core.reset(seed)
        except Exception:
            logger.exception("[env.reset] StaffingCore.reset() FAILED  seed=%s", seed)
            raise
        self._passive_streak = 0
        self._last_tool = ""
        self._last_tool_failed = False
        self._last_tool_reward = 0.0
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
        "get_market_demand", "get_financial_summary", "get_candidate_types",
        "_override_cash", "_override_satisfaction",
    })

    def step(self, action: Any, timeout_s: float | None = None, **kwargs) -> Observation:
        """Route MCP tool calls + capture rewards.

        ListToolsAction → pass through raw ListToolsObservation (no tick).
        GET CallToolAction → run tool only, no tick (pure observation).
        EXECUTE CallToolAction → run tool + step increment (via advance_week).

        REWARD FLOW:
          MCPEnvironment.step() returns obs.reward == 0 always (framework
          does not extract reward from tool result dicts). Each tool wrapper
          stores its raw reward in self._last_tool_reward before stripping
          the "reward" key so the agent cannot see it. We read that stored
          value here and add penalties to form total_reward.
        """
        # Tool discovery: no tick, no reward
        if isinstance(action, ListToolsAction):
            return super().step(action, timeout_s=timeout_s, **kwargs)

        # Reset per-step reward accumulator before the tool runs
        self._last_tool_reward = 0.0

        # Strip hallucinated args the LLM adds to tools that don't accept them.
        # FastMCP uses Pydantic strict validation — any unexpected kwarg raises
        # ValidationError which bypasses _last_tool_reward and gives reward=0.
        tool_name_for_strip = getattr(action, "tool_name", "")
        if tool_name_for_strip in self._TOOL_VALID_ARGS and hasattr(action, "arguments"):
            valid = self._TOOL_VALID_ARGS[tool_name_for_strip]
            bad = {k for k in action.arguments if k not in valid}
            if bad:
                logger.warning(
                    "[env.step] Stripping hallucinated args from %s: %s",
                    tool_name_for_strip, sorted(bad),
                )
                action.arguments = {k: v for k, v in action.arguments.items() if k in valid}

        # Let MCPEnvironment route and execute the tool call
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Track whether this call failed (so we can waive the repeat penalty on retry)
        tr = obs.tool_result if hasattr(obs, "tool_result") else {}
        if isinstance(tr, dict):
            self._last_tool_failed = not tr.get("success", True)
        else:
            self._last_tool_failed = False

        # Sync state with core after action (tool might have changed cash/etc)
        self._state.step_count = self.core.step_count
        self._state.cash = self.core.cash
        self._state.revenue = self.core.revenue
        self._state.costs = self.core.costs

        # Capture the tool's own reward (set by each wrapper into _last_tool_reward)
        tool_reward = self._last_tool_reward

        # Repeat-call penalty: skip if the PREVIOUS call of the same tool failed
        # (retrying after a genuine failure is correct behaviour, not spam).
        tool_name = action.tool_name if hasattr(action, "tool_name") else ""
        repeat_penalty = 0.0
        if (tool_name and tool_name == self._last_tool
                and tool_name not in self._GET_TOOLS
                and not self._last_tool_failed):
            repeat_penalty = self._config.repeat_call_penalty
        self._last_tool = tool_name

        # Passive-streak penalty: same — shaping only, not a business cost
        if tool_name in self._PASSIVE_TOOLS:
            self._passive_streak += 1
        else:
            self._passive_streak = 0

        passive_penalty = 0.0
        if self._passive_streak > self._config.passive_streak_threshold:
            passive_penalty = self._config.passive_streak_penalty

        total_reward = tool_reward + passive_penalty + repeat_penalty
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

        logger.debug(
            "[env.step] tool=%s  tool_rew=%+.2f  passive=%+.2f  repeat=%+.2f  "
            "total=%+.2f  cash=$%s  profit=$%s  done=%s",
            tool_name, tool_reward, passive_penalty, repeat_penalty, total_reward,
            f"{self._state.cash:,.0f}",
            f"{self._state.revenue - self._state.costs:,.0f}",
            done,
        )
        if done:
            logger.warning(
                "[env.step] Episode DONE  reason=%s  cash=$%s  step=%s",
                "bankrupt" if self._state.cash < 0 else "max_steps",
                f"{self._state.cash:,.0f}", self._state.step_count,
            )

        # Unwrap CallToolObservation → plain dict (strip reward key so agent never sees it)
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

        # Ensure reward never leaks into agent-visible tool_result
        if isinstance(tool_result, dict):
            tool_result.pop("reward", None)

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

        # ---- GET tools (reward always 0, no state mutation) ----

        @mcp.tool()
        def get_agency_state() -> dict:
            return env.core.tool_get_agency_state()["state"]

        @mcp.tool()
        def get_client_state(client_id: str = "") -> dict:
            res = env.core.tool_get_client_state(client_id)
            if "error" in res:
                return res
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
            if "error" in res:
                return {"error": res["error"]}
            return dict(res["project"])

        @mcp.tool()
        def get_candidate_profile(candidate_id: str) -> dict:
            res = env.core.tool_get_candidate_profile(candidate_id)
            if "error" in res:
                return {"error": res["error"]}
            return dict(res["candidate"])

        @mcp.tool()
        def get_market_demand(sector: str = "", developer_type: str = "") -> dict:
            """See which developer roles are currently most requested by clients."""
            r = env.core.tool_get_market_demand()
            return {"demand_by_type": r["demand_by_type"], "total_open_slots": r["total_open_slots"]}

        @mcp.tool()
        def get_financial_summary() -> dict:
            return env.core.tool_get_financial_summary()["summary"]

        # ---- EXECUTE tools — capture reward before stripping ----

        @mcp.tool()
        def find_available_projects() -> dict:
            r = env.core.tool_find_available_projects()
            return {"projects": r["projects"], "count": r["count"]}

        @mcp.tool()
        def confirm_project(project_id: str) -> dict:
            res = env.core.tool_confirm_project(project_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                open_projects = [
                    {"project_id": p.project_id,
                     "roles": [{"role_id": ro.role_id, "needs_type": ro.developer_type,
                                "min_skill": ro.min_skill_score, "bill_rate": ro.bill_rate_weekly}
                               for ro in p.roles if not ro.is_filled]}
                    for cl in env.core.clients for p in cl.projects
                    if p.fill_status != "SEALED"
                ]
                res["reason"] = f"Project '{project_id}' not found or already sealed."
                res["open_projects"] = open_projects[:6]
                res["fix"] = (
                    "Call find_available_projects() to get valid project_ids, "
                    "then confirm_project(project_id=<valid_id>)."
                )
            else:
                # On success, show the roles that now need to be filled
                project = next(
                    (p for cl in env.core.clients for p in cl.projects if p.project_id == project_id), None
                )
                if project:
                    open_roles = [{"role_id": ro.role_id, "needs_type": ro.developer_type,
                                   "min_skill": ro.min_skill_score, "bill_rate": ro.bill_rate_weekly}
                                  for ro in project.roles if not ro.is_filled]
                    res["roles_to_fill"] = open_roles
                    res["next_step"] = (
                        f"Now match a hired candidate to one of these roles: {open_roles[:3]}. "
                        f"Call match_candidate_to_project(candidate_id=<id>, "
                        f"project_id=\"{project_id}\", role_id=<role_id>)."
                    )
            return res

        @mcp.tool()
        def get_candidate_types() -> dict:
            """Return the valid developer_type and seniority_level values accepted by find_candidate, plus score ranges."""
            return env.core.tool_get_candidate_types()

        @mcp.tool()
        def find_candidate(
            developer_type: str = "",
            seniority_level: str = "",
            min_skill_score: float = 0.0,
            min_composite_rating: float = 0.0,
        ) -> dict:
            """Search the market. All filters optional and combinable."""
            r = env.core.tool_find_candidate(
                developer_type, seniority_level, min_skill_score, min_composite_rating
            )
            result = {"candidates": r["candidates"], "count": r["count"]}
            if r["count"] == 0:
                from collections import Counter
                type_counts = Counter(c.developer_type for c in env.core.market)
                result["no_results_reason"] = (
                    f"No candidates match filters: developer_type='{developer_type}' "
                    f"seniority='{seniority_level}' min_skill={min_skill_score}."
                )
                result["available_types_in_market"] = dict(type_counts)
                result["fix"] = "Adjust your filters or call find_candidate() with no filters."
            else:
                # Prominently remind the model of the required two-step before hiring
                ids = [c["id"] for c in result["candidates"][:5]]
                result["REQUIRED_NEXT_STEP"] = (
                    f"You MUST call interview_candidate(candidate_id=<id>) BEFORE hire_candidate. "
                    f"Valid candidate IDs from this search: {ids}. "
                    f"Example: interview_candidate(candidate_id=\"{ids[0]}\")"
                )
            return result

        @mcp.tool()
        def interview_candidate(candidate_id: str) -> dict:
            """Screen a candidate: costs $500, reveals skill + salary expectation."""
            res = env.core.tool_interview_candidate(candidate_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            res.pop("result", None)
            if not res.get("success", True):
                # Show available market candidates so model can pick a real one
                market = [
                    {"id": c.id, "type": c.developer_type, "seniority": c.seniority_level}
                    for c in env.core.market[:10]
                ]
                res["reason"] = (
                    f"Candidate '{candidate_id}' is not in the market "
                    f"(already hired, placed, or ID is wrong)."
                )
                res["available_in_market"] = market
                res["fix"] = "Use one of the candidate IDs from 'available_in_market' above."
            return res

        @mcp.tool()
        def hire_candidate(candidate_id: str) -> dict:
            """Hire an interviewed candidate: costs $2,000 onboarding."""
            res = env.core.tool_hire_candidate(candidate_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))

            if not res.get("success", True):
                # Check if candidate is in market (needs interview) or truly gone
                in_market = next((c for c in env.core.market if c.id == candidate_id), None)
                if in_market:
                    res["reason"] = (
                        f"Candidate '{candidate_id}' IS in the market but has NOT been interviewed yet. "
                        f"You MUST call interview_candidate(candidate_id=\"{candidate_id}\") FIRST, "
                        f"then hire_candidate(candidate_id=\"{candidate_id}\")."
                    )
                    res["fix"] = f"Call interview_candidate(candidate_id=\"{candidate_id}\") now."
                else:
                    market_now = [{"id": c.id, "type": c.developer_type, "seniority": c.seniority_level}
                                  for c in env.core.market[:8]]
                    res["reason"] = (
                        f"Candidate '{candidate_id}' no longer exists (left market or wrong ID). "
                        f"Current market has {len(env.core.market)} candidates."
                    )
                    res["current_market_ids"] = market_now
                    res["fix"] = "Call find_candidate() to get current IDs, then interview_candidate(), then hire_candidate()."
                return res

            # Attach compatible open roles so the model can match immediately
            if res.get("hired"):
                hired_type  = res.get("developer_type", "")
                hired_skill = res.get("composite_rating", 0.0)
                compatible = []
                for cl in env.core.clients:
                    for proj in cl.projects:
                        if proj.fill_status == "SEALED":
                            continue
                        for role in proj.roles:
                            if (not role.is_filled
                                    and role.developer_type == hired_type
                                    and hired_skill >= role.min_skill_score):
                                compatible.append({
                                    "project_id": proj.project_id,
                                    "role_id":    role.role_id,
                                    "needs_type": role.developer_type,
                                    "min_skill":  role.min_skill_score,
                                    "bill_rate_weekly": role.bill_rate_weekly,
                                })
                if compatible:
                    res["next_step"] = (
                        f"Call confirm_project then match_candidate_to_project using one of these "
                        f"compatible '{hired_type}' roles: {compatible[:4]}"
                    )
                else:
                    res["next_step"] = (
                        f"No open '{hired_type}' roles right now. "
                        "Call get_client_state to find new projects or advance_week."
                    )
            return res

        @mcp.tool()
        def negotiate_salary(candidate_id: str, offer_weekly: float) -> dict:
            res = env.core.tool_negotiate_salary(candidate_id, offer_weekly)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                in_market = next((c for c in env.core.market if c.id == candidate_id), None)
                in_roster = env.core.candidates.get(candidate_id)
                if in_market:
                    res["reason"] = f"Candidate '{candidate_id}' is in the market but must be interviewed first."
                    res["fix"] = f"Call interview_candidate(candidate_id=\"{candidate_id}\") first."
                elif not in_roster:
                    market = [{"id": c.id, "type": c.developer_type} for c in env.core.market[:6]]
                    res["reason"] = f"Candidate '{candidate_id}' not found anywhere."
                    res["current_market"] = market
                    res["fix"] = "Call find_candidate() to get fresh IDs."
                else:
                    res["reason"] = f"Cannot negotiate with candidate '{candidate_id}' in status '{in_roster.status}'."
                    res["fix"] = "Negotiate only with candidates who have been interviewed (status=in_pipeline)."
            return res

        @mcp.tool()
        def match_candidate_to_project(candidate_id: str, project_id: str, role_id: str) -> dict:
            """Place a candidate on a role. Speed bonus if project sealed within 2 weeks."""
            res = env.core.tool_match_candidate_to_project(candidate_id, project_id, role_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                err = res.get("error", "")
                # Candidate not in hired status
                if "hired" in err.lower() or "status" in err.lower() or "available" in err.lower():
                    hired = [{"id": c.id, "type": c.developer_type, "status": c.status}
                             for c in env.core.candidates.values()
                             if c.status == "hired"]
                    in_market = next((c for c in env.core.market if c.id == candidate_id), None)
                    if in_market:
                        res["reason"] = (
                            f"Candidate '{candidate_id}' is still in the market (not yet hired). "
                            f"Workflow: find_candidate → interview_candidate → hire_candidate → match."
                        )
                        res["fix"] = f"Call interview_candidate(candidate_id=\"{candidate_id}\") then hire_candidate."
                    else:
                        res["reason"] = f"Candidate '{candidate_id}' is not in 'hired' status."
                        res["your_hired_candidates"] = hired
                        res["fix"] = "Only match candidates with status='hired'. Use an ID from 'your_hired_candidates'."
                # Type/skill mismatch: show what the candidate has vs what the role needs
                elif "type mismatch" in err.lower() or "insufficient skill" in err.lower() or "type mismatch" in err.upper():
                    c_type  = res.get("candidate_type", "?")
                    r_type  = res.get("role_type", "?")
                    c_skill = res.get("candidate_skill", "?")
                    r_skill = res.get("role_min_skill", "?")
                    # Suggest open roles that DO match the candidate's type
                    matching_roles = [
                        {"project_id": p.project_id, "role_id": ro.role_id,
                         "needs": ro.developer_type, "min_skill": ro.min_skill_score,
                         "bill_rate": ro.bill_rate_weekly}
                        for cl in env.core.clients for p in cl.projects
                        if p.fill_status != "SEALED"
                        for ro in p.roles
                        if not ro.is_filled and ro.developer_type == c_type
                    ]
                    res["reason"] = (
                        f"Your candidate '{candidate_id}' is type='{c_type}' skill={c_skill}. "
                        f"Role '{role_id}' needs type='{r_type}' min_skill={r_skill}. Types must match exactly."
                    )
                    res["roles_matching_your_candidate"] = matching_roles[:5]
                    res["fix"] = (
                        f"Use a role from 'roles_matching_your_candidate' that needs '{c_type}'. "
                        f"If none listed, call find_candidate(developer_type='{r_type}') to hire the right type."
                    )
                # Project or role not found
                elif "not found" in err.lower() or "invalid" in err.lower():
                    open_projects = [
                        {"project_id": p.project_id, "client_id": cl.client_id,
                         "roles": [{"role_id": ro.role_id, "needs": ro.developer_type}
                                   for ro in p.roles if not ro.is_filled]}
                        for cl in env.core.clients for p in cl.projects
                        if p.fill_status != "SEALED"
                    ]
                    res["reason"] = f"project_id='{project_id}' or role_id='{role_id}' does not exist."
                    res["open_projects"] = open_projects[:5]
                    res["fix"] = "Use project_id and role_id from 'open_projects' above — never invent IDs."
            return res

        @mcp.tool()
        def let_go_candidate(candidate_id: str) -> dict:
            """Release a candidate: costs 2× weekly salary as severance."""
            res = env.core.tool_let_go_candidate(candidate_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                hired = [{"id": c.id, "type": c.developer_type, "status": c.status,
                          "salary_weekly": round(c.salary_weekly, 2)}
                         for c in env.core.candidates.values()
                         if c.status in ("hired", "in_pipeline")]
                res["reason"] = f"Candidate '{candidate_id}' not found in your roster."
                res["your_hired_candidates"] = hired
                res["fix"] = "Use an ID from 'your_hired_candidates'. Call get_candidate_state() to refresh."
            return res

        @mcp.tool()
        def request_project_extension(project_id: str) -> dict:
            res = env.core.tool_request_project_extension(project_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                open_projects = [{"project_id": p.project_id, "deadline_week": p.deadline_week}
                                 for cl in env.core.clients for p in cl.projects
                                 if p.fill_status != "SEALED"]
                res["reason"] = f"Project '{project_id}' not found or not extendable."
                res["open_projects"] = open_projects[:6]
                res["fix"] = "Use a project_id from 'open_projects'. Call find_available_projects() to refresh."
            return res

        @mcp.tool()
        def pass_on_project(project_id: str) -> dict:
            res = env.core.tool_pass_on_project(project_id)
            env._last_tool_reward = float(res.pop("reward", 0.0))
            if not res.get("success", True):
                open_projects = [{"project_id": p.project_id, "fill_status": p.fill_status}
                                 for cl in env.core.clients for p in cl.projects
                                 if p.fill_status != "SEALED"]
                res["reason"] = f"Project '{project_id}' not found."
                res["open_projects"] = open_projects[:6]
                res["fix"] = "Use a project_id from 'open_projects'. Call find_available_projects() to refresh."
            return res

        @mcp.tool()
        def advance_week() -> dict:
            """Advance simulation to the next week (ticks world, pays costs/revenue).

            This is where the major reward signal arrives:
              +margin_weekly per placed candidate
              -salary_weekly per benched candidate
              -expiry_penalty per expired unfilled project role
              -client_ltv_estimate if client satisfaction < churn_threshold
            """
            res = env.core.tool_advance_week()
            # REWARD FIX: capture world_tick reward (billing margins, bench burn,
            # expiry penalties, churn penalties) before stripping from agent view.
            env._last_tool_reward = float(res.pop("reward", 0.0))
            return res

        # ---- Admin / UI-only tools (no episode reward) ----

        @mcp.tool()
        def _override_cash(amount: float) -> dict:
            old = env.core.cash
            env.core.cash = float(amount)
            env._state.cash = env.core.cash
            return {"success": True, "old_cash": round(old, 2), "new_cash": round(env.core.cash, 2)}

        @mcp.tool()
        def _override_satisfaction(client_id: str, score: float) -> dict:
            client = next((c for c in env.core.clients if c.client_id == client_id), None)
            if not client:
                return {"success": False, "error": f"Client {client_id} not found"}
            old = client.satisfaction_score
            client.satisfaction_score = max(0.0, min(1.0, float(score)))
            client.churn_risk = client.satisfaction_score < env._config.churn_threshold
            return {
                "success": True,
                "client_id": client_id,
                "old_score": round(old, 3),
                "new_score": round(client.satisfaction_score, 3),
                "churn_risk": client.churn_risk,
            }

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
