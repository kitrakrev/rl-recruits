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
from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
from openenv.core.env_server.types import Observation, State

from env.config import Config
from env.models import Candidate, Role, Project, Client
from env.llm import LLMRouter
from env.simulation import (
    generate_client,
    replenish_market,
    tick_project_arrivals,
    tick_project_deadlines,
    tick_candidate_patience,
    tick_contracts,
    compute_match_score,
)


# ---------------------------------------------------------------------------
# Environment State (Pydantic, persisted across steps)
# ---------------------------------------------------------------------------

class StaffingState(State):
    """Full serialisable state — returned from GET /state."""
    episode_id: str = ""
    step_count: int = 0
    cash: float = 50_000.0
    revenue: float = 0.0
    costs: float = 0.0
    num_placed: int = 0
    num_hired: int = 0
    num_benched: int = 0
    avg_satisfaction: float = 0.75
    cumulative_reward: float = 0.0
    done: bool = False


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

    def __init__(self, config: Config | None = None):
        self._config = config or Config(
            llm_mode=os.getenv("LLM_MODE", "stub"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        self._llm = LLMRouter(self._config)

        # Runtime state (reset on each episode)
        self._state = StaffingState()
        self._clients: list[Client] = []
        self._candidates: dict[str, Candidate] = {}
        self._market: list[Candidate] = []
        self._pending_rewards: list[float] = []
        self._time_to_place: list[float] = []
        self._hire_step: dict[str, int] = {}
        self._expired_projects: list[str] = []

        import random
        self._rng = random.Random()

        # Build FastMCP server and register all tools
        mcp = FastMCP("staffing_agency")
        self._register_tools(mcp)

        super().__init__(mcp)

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> Observation:
        import random
        self._rng = random.Random(seed)

        episode_id = episode_id or str(uuid4())
        self._state = StaffingState(
            episode_id=episode_id,
            step_count=0,
            cash=self._config.seed_capital,
            revenue=0.0,
            costs=0.0,
            done=False,
        )
        self._clients = [
            generate_client(i, self._config, self._rng)
            for i in range(self._config.num_clients)
        ]
        self._candidates = {}
        self._market = []
        self._pending_rewards = []
        self._time_to_place = []
        self._hire_step = {}
        self._expired_projects = []

        replenish_market(self._market, self._config, self._rng)
        # Seed initial projects so agent has something to work with
        tick_project_arrivals(self._clients, self._config, self._rng)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "episode_id": episode_id,
                "message": (
                    "Welcome to Staffing Agency RL. You are the agency CEO. "
                    "Recruit candidates, fill client projects, maximise profit over 52 weeks. "
                    "Use list_tools() to see all available actions."
                ),
                "state": self._snapshot(),
            },
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

        # GET tools: read-only, no tick, no step count increment
        if isinstance(action, CallToolAction) and action.tool_name in self._GET_TOOLS:
            obs = super().step(action, timeout_s=timeout_s, **kwargs)
            tool_result = None
            if hasattr(obs, "result") and obs.result is not None:
                r = obs.result
                tool_result = r.data if hasattr(r, "data") else r
            return Observation(
                done=self._state.done,
                reward=0.0,
                metadata={
                    "step": self._state.step_count,
                    "cash": round(self._state.cash, 2),
                    "profit": round(self._state.revenue - self._state.costs, 2),
                    "cumulative_reward": round(self._state.cumulative_reward, 4),
                    "tool_result": tool_result,
                },
            )

        self._state.step_count += 1

        # Let MCPEnvironment route EXECUTE CallToolAction
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # World tick: advance time, compute P&L reward
        step_reward = self._world_tick()
        total_reward = (obs.reward or 0.0) + step_reward / self._config.reward_scale

        self._state.cumulative_reward += total_reward

        done = (
            self._state.cash < 0
            or self._state.step_count >= self._config.episode_steps
        )
        self._state.done = done
        self._state.num_hired = sum(
            1 for c in self._candidates.values() if c.status in ("hired", "placed")
        )
        self._state.num_placed = sum(
            1 for c in self._candidates.values() if c.status == "placed"
        )
        self._state.num_benched = sum(
            1 for c in self._candidates.values() if c.status == "hired"
        )
        active = [cl for cl in self._clients if not cl.churn_risk]
        self._state.avg_satisfaction = (
            sum(cl.satisfaction_score for cl in active) / len(active) if active else 0.0
        )

        # Unwrap CallToolObservation → plain dict
        # FastMCP returns CallToolResult; the parsed dict is in .data
        tool_result = None
        if hasattr(obs, "result") and obs.result is not None:
            r = obs.result
            if hasattr(r, "data"):
                tool_result = r.data          # structured dict
            elif hasattr(r, "structured_content"):
                tool_result = r.structured_content
            else:
                tool_result = r
        elif hasattr(obs, "metadata"):
            tool_result = obs.metadata

        return Observation(
            done=done,
            reward=total_reward,
            metadata={
                "step": self._state.step_count,
                "cash": round(self._state.cash, 2),
                "profit": round(self._state.revenue - self._state.costs, 2),
                "cumulative_reward": round(self._state.cumulative_reward, 4),
                "tool_result": tool_result,
            },
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
    # World tick (runs every step)
    # ------------------------------------------------------------------

    def _world_tick(self) -> float:
        reward = 0.0
        cfg = self._config

        # 1. Billing: placed candidates generate margin
        for c in self._candidates.values():
            if c.status == "placed":
                self._state.cash += c.client_rate_weekly
                self._state.revenue += c.client_rate_weekly
                self._state.costs += c.salary_weekly
                self._state.cash -= c.salary_weekly
                reward += c.margin_weekly

        # 2. Bench burn: hired-but-unplaced costs salary
        for c in self._candidates.values():
            if c.status == "hired":
                self._state.cash -= c.salary_weekly
                self._state.costs += c.salary_weekly
                reward -= c.salary_weekly

        # 3. Project arrivals (Poisson)
        tick_project_arrivals(self._clients, cfg, self._rng)

        # 4. Project deadline countdown + expiry penalty
        expired = tick_project_deadlines(self._clients, self._llm)
        for project, client in expired:
            for role in project.roles:
                unfilled = role.headcount - role.filled_count
                if unfilled > 0:
                    penalty = unfilled * (85_000 / 52) * 1.25
                    reward -= penalty
                    self._state.costs += penalty
            self._expired_projects.append(project.project_id)

        # 5. Contract completions → return to bench
        returning = tick_contracts(list(self._candidates.values()))
        for c in returning:
            c.status = "hired"
            c.assigned_project = None
            c.assigned_role = None
            c.contract_weeks_left = None
            c.weeks_on_bench = 0
            # Remove from role assignments
            for client in self._clients:
                for project in client.projects:
                    for role in project.roles:
                        if c.id in role.assigned:
                            role.assigned.remove(c.id)
                            role.filled_count = max(0, role.filled_count - 1)
                            project.update_fill_status()

        # 6. Candidate patience decay
        agency_ctx = {
            "step": self._state.step_count,
            "open_roles": self._count_open_roles(),
        }
        leavers = tick_candidate_patience(
            list(self._candidates.values()), self._llm, agency_ctx
        )
        for c in leavers:
            self._candidates.pop(c.id, None)

        # 7. Client churn penalty
        for client in self._clients:
            if (
                client.satisfaction_score < cfg.churn_threshold
                and not client.churn_risk
            ):
                client.churn_risk = True
                reward -= cfg.client_ltv_estimate
                self._state.costs += cfg.client_ltv_estimate

        # 8. Replenish market
        replenish_market(self._market, cfg, self._rng)

        return reward

    # ------------------------------------------------------------------
    # Tool registration via FastMCP
    # ------------------------------------------------------------------

    def _register_tools(self, mcp: FastMCP) -> None:
        env = self  # closure reference

        # ── GET TOOLS ──────────────────────────────────────────────────

        @mcp.tool()
        def get_agency_state() -> dict:
            """Get full agency financial snapshot: cash, revenue, costs, profit, burn rate, runway."""
            hired = [c for c in env._candidates.values() if c.status in ("hired", "placed")]
            placed = [c for c in env._candidates.values() if c.status == "placed"]
            benched = [c for c in env._candidates.values() if c.status == "hired"]
            pipeline = [c for c in env._candidates.values() if c.status == "in_pipeline"]
            burn = sum(c.salary_weekly for c in hired)
            runway = (env._state.cash / burn) if burn > 0 else 999.0
            placement_rate = len(placed) / len(hired) if hired else 0.0
            import numpy as np
            avg_ttp = float(np.mean(env._time_to_place)) if env._time_to_place else 0.0
            return {
                "cash_balance": round(env._state.cash, 2),
                "current_revenue": round(env._state.revenue, 2),
                "current_costs": round(env._state.costs, 2),
                "current_profit": round(env._state.revenue - env._state.costs, 2),
                "num_candidates_hired": len(hired),
                "num_candidates_placed": len(placed),
                "num_candidates_benched": len(benched),
                "num_candidates_in_interview": len(pipeline),
                "placement_rate": round(placement_rate, 3),
                "avg_time_to_place": round(avg_ttp, 2),
                "burn_rate": round(burn, 2),
                "cash_runway_weeks": round(min(runway, 999.0), 2),
                "step": env._state.step_count,
                "episode_steps": env._config.episode_steps,
            }

        @mcp.tool()
        def get_client_state(client_id: str = "") -> dict:
            """Get state for one client (pass client_id) or all clients (leave blank)."""
            if client_id:
                c = env._find_client(client_id)
                return c.to_dict() if c else {"error": f"Client {client_id} not found"}
            return {"clients": [c.to_dict() for c in env._clients]}

        @mcp.tool()
        def get_candidate_state() -> dict:
            """Get full candidate pool state: pipeline, bench, churn risks."""
            cands = list(env._candidates.values())
            return {
                "num_in_market": len(env._market),
                "num_in_pipeline": sum(1 for c in cands if c.status == "in_pipeline"),
                "num_pending_hire": sum(1 for c in cands if c.status == "pending_hire"),
                "num_hired": sum(1 for c in cands if c.status in ("hired", "placed")),
                "num_placed": sum(1 for c in cands if c.status == "placed"),
                "churn_risk_ids": [c.id for c in cands if c.patience_remaining <= 2],
                "candidates": [c.to_dict() for c in cands],
            }

        @mcp.tool()
        def get_project_details(project_id: str) -> dict:
            """Get full details of a specific project including roles and fill status."""
            p = env._find_project(project_id)
            return p.to_dict() if p else {"error": f"Project {project_id} not found"}

        @mcp.tool()
        def get_candidate_profile(candidate_id: str) -> dict:
            """Get full profile of a specific candidate."""
            c = env._candidates.get(candidate_id)
            return c.to_dict() if c else {"error": f"Candidate {candidate_id} not found"}

        @mcp.tool()
        def get_market_demand() -> dict:
            """Get current demand by developer type across all open project roles."""
            demand = {dt: 0 for dt in env._config.developer_types}
            for client in env._clients:
                if client.churn_risk:
                    continue
                for project in client.projects:
                    for role in project.roles:
                        if not role.is_filled:
                            demand[role.developer_type] += (role.headcount - role.filled_count)
            return {"demand_by_type": demand, "total_open_slots": sum(demand.values())}

        @mcp.tool()
        def get_financial_summary() -> dict:
            """Get P&L summary, burn rate, and cash runway."""
            hired = [c for c in env._candidates.values() if c.status in ("hired", "placed")]
            burn = sum(c.salary_weekly for c in hired)
            runway = (env._state.cash / burn) if burn > 0 else 999.0
            return {
                "cash_balance": round(env._state.cash, 2),
                "revenue": round(env._state.revenue, 2),
                "costs": round(env._state.costs, 2),
                "profit": round(env._state.revenue - env._state.costs, 2),
                "burn_rate": round(burn, 2),
                "cash_runway_weeks": round(min(runway, 999.0), 2),
            }

        # ── EXECUTE TOOLS ──────────────────────────────────────────────

        @mcp.tool()
        def find_available_projects() -> dict:
            """Discover all open/partial projects across all active clients."""
            projects = []
            for client in env._clients:
                if client.churn_risk:
                    continue
                for p in client.projects:
                    if p.fill_status != "SEALED":
                        d = p.to_dict()
                        d["client_industry"] = client.industry
                        d["client_satisfaction"] = client.satisfaction_score
                        projects.append(d)
            return {"projects": projects, "count": len(projects)}

        @mcp.tool()
        def confirm_project(project_id: str) -> dict:
            """Commit to filling a project. Signals intent to client (small satisfaction boost)."""
            project = env._find_project(project_id)
            if not project:
                return {"success": False, "error": f"Project {project_id} not found"}
            project.confirmed = True
            client = env._find_client(project.client_id)
            if client:
                event = {"type": "project_confirmed", "project_id": project_id}
                result = env._llm.client_satisfaction(client, event, client.event_history)
                client.satisfaction_score = result.new_score
                client.churn_risk = result.churn_risk
                client.event_history.append(event)
                return {
                    "success": True,
                    "project_id": project_id,
                    "client_satisfaction": round(result.new_score, 3),
                }
            return {"success": True, "project_id": project_id}

        @mcp.tool()
        def find_candidate(developer_type: str = "") -> dict:
            """Search the market for available candidates. Filter by developer_type or leave blank for all."""
            if developer_type:
                found = [c for c in env._market if c.developer_type == developer_type]
            else:
                found = env._market[:]
            return {
                "candidates": [c.to_dict() for c in found],
                "count": len(found),
            }

        @mcp.tool()
        def interview_candidate(candidate_id: str) -> dict:
            """
            Interview a candidate from the market. Sets base_rating (1-5).
            Moves candidate from market to your pipeline.
            """
            c = env._find_in_market(candidate_id)
            if not c:
                return {"success": False, "error": f"Candidate {candidate_id} not in market"}
            job_desc = f"{c.developer_type} {c.seniority_level}"
            result = env._llm.interview(c, job_desc)
            c.base_rating = result.base_rating
            c.technical_score = result.technical_score
            c.communication = result.communication
            c.culture_fit = result.culture_fit
            c.red_flags = result.red_flags
            c.interview_summary = result.summary
            c.status = "in_pipeline"
            env._market.remove(c)
            env._candidates[c.id] = c
            return {
                "success": True,
                "candidate_id": c.id,
                "base_rating": result.base_rating,
                "proceed": result.proceed,
                "summary": result.summary,
                "red_flags": result.red_flags,
                "technical_score": result.technical_score,
            }

        @mcp.tool()
        def hire_candidate(candidate_id: str) -> dict:
            """
            Hire a candidate from pipeline. Costs $2,000 onboarding.
            Candidate starts accruing weekly salary immediately (bench cost).
            """
            c = env._candidates.get(candidate_id)
            if not c:
                return {"success": False, "error": f"Candidate {candidate_id} not found"}
            if c.status not in ("in_pipeline", "pending_hire"):
                return {"success": False, "error": f"Cannot hire: status is '{c.status}'"}
            cfg = env._config
            env._state.cash -= cfg.onboarding_cost
            env._state.costs += cfg.onboarding_cost
            composite = round(0.4 * c.base_rating + 0.6 * 3.0, 2)
            c.composite_rating = composite
            c.salary_weekly, c.client_rate_weekly = cfg.salary_from_rating(composite)
            c.margin_weekly = c.client_rate_weekly - c.salary_weekly
            c.status = "hired"
            env._hire_step[c.id] = env._state.step_count
            return {
                "success": True,
                "candidate_id": c.id,
                "developer_type": c.developer_type,
                "seniority": c.seniority_level,
                "composite_rating": composite,
                "salary_weekly": round(c.salary_weekly, 2),
                "onboarding_cost": cfg.onboarding_cost,
                "bench_burn_per_week": round(c.salary_weekly, 2),
                "break_even_weeks": round(cfg.onboarding_cost / c.margin_weekly, 1),
            }

        @mcp.tool()
        def negotiate_salary(candidate_id: str, offer_weekly: float) -> dict:
            """
            Negotiate salary with a pipeline candidate.
            If accepted, candidate moves to pending_hire. Provide weekly $ offer.
            """
            c = env._candidates.get(candidate_id)
            if not c:
                return {"success": False, "error": f"Candidate {candidate_id} not found"}
            market_ctx = {"step": env._state.step_count, "market_size": len(env._market)}
            result = env._llm.salary_negotiation(c, offer_weekly, market_ctx)
            c.patience_remaining = max(0, c.patience_remaining + result.patience_impact)
            if result.accepted:
                c.salary_expectation = offer_weekly
                c.status = "pending_hire"
            return {
                "success": True,
                "accepted": result.accepted,
                "counter_offer": result.counter_offer,
                "reason": result.acceptance_reason,
                "candidate_expectation": c.salary_expectation,
            }

        @mcp.tool()
        def match_candidate_to_project(
            candidate_id: str, project_id: str, role_id: str
        ) -> dict:
            """
            Place a hired candidate into a specific project role.
            Revenue only starts when ALL roles in the project are filled (project SEALED).
            Returns match_score, fit_rating, and whether the project is now SEALED.
            """
            c = env._candidates.get(candidate_id)
            if not c or c.status != "hired":
                return {"success": False, "error": "Candidate must be in 'hired' status"}
            project = env._find_project(project_id)
            if not project:
                return {"success": False, "error": f"Project {project_id} not found"}
            role = next((r for r in project.roles if r.role_id == role_id), None)
            if not role:
                return {"success": False, "error": f"Role {role_id} not found"}
            if role.is_filled:
                return {"success": False, "error": "Role is already fully filled"}

            match_score = compute_match_score(c, role, env._config)
            if match_score == 0.0:
                return {
                    "success": False,
                    "error": "Illegal assignment — type mismatch or insufficient skill",
                    "candidate_type": c.developer_type,
                    "role_type": role.developer_type,
                    "candidate_skill": c.skill_score,
                    "role_min_skill": role.min_skill_score,
                }

            # LLM fit assessment
            client = env._find_client(project.client_id)
            project_ctx = {
                "client_industry": client.industry if client else "unknown",
                "project_id": project_id,
            }
            fit = env._llm.project_fit(c, role, project_ctx)

            # Update candidate
            c.project_fit_rating = fit.project_fit_rating
            c.composite_rating = fit.composite_rating
            c.salary_weekly, c.client_rate_weekly = env._config.salary_from_rating(c.composite_rating)
            c.margin_weekly = c.client_rate_weekly - c.salary_weekly
            c.status = "placed"
            c.assigned_project = project_id
            c.assigned_role = role_id
            c.contract_weeks_left = env._config.contract_duration
            c.weeks_on_bench = 0

            # Update role
            role.filled_count += 1
            role.assigned.append(candidate_id)
            project.update_fill_status()

            sealed_now = project.fill_status == "SEALED"
            tool_reward = 0.0

            # Client satisfaction
            if client:
                evt_type = "project_sealed" if sealed_now else (
                    "adjacent_match" if match_score < 1.0 else "partial_fill"
                )
                event = {
                    "type": evt_type,
                    "project_id": project_id,
                    "match_score": match_score,
                }
                sat = env._llm.client_satisfaction(client, event, client.event_history)
                client.satisfaction_score = sat.new_score
                client.churn_risk = sat.churn_risk
                client.event_history.append(event)
                if sealed_now:
                    client.num_projects_filled += 1
                    total_hc = sum(r.headcount for r in project.roles)
                    project.weekly_revenue = client.contracted_rate * total_hc

            # Speed bonus
            if sealed_now and project.confirmed:
                original_deadline = env._config.t_deadline_max
                weeks_taken = original_deadline - project.deadline_remaining
                if weeks_taken <= 2:
                    speed_bonus = c.margin_weekly * 0.1
                    tool_reward += speed_bonus / env._config.reward_scale

            # Track time-to-place
            if c.id in env._hire_step:
                ttp = env._state.step_count - env._hire_step[c.id]
                env._time_to_place.append(float(ttp))

            return {
                "success": True,
                "match_score": match_score,
                "composite_rating": c.composite_rating,
                "project_fit_rating": fit.project_fit_rating,
                "fit_rationale": fit.fit_rationale,
                "project_sealed": sealed_now,
                "weekly_margin": round(c.margin_weekly, 2),
                "speed_bonus": round(tool_reward * env._config.reward_scale, 2),
                "risk_flags": fit.risk_flags,
            }

        @mcp.tool()
        def let_go_candidate(candidate_id: str) -> dict:
            """
            Remove a candidate from payroll. Costs 2× weekly salary as severance.
            Use to trim bench and stop salary burn.
            """
            c = env._candidates.get(candidate_id)
            if not c:
                return {"success": False, "error": f"Candidate {candidate_id} not found"}
            severance = c.salary_weekly * env._config.severance_weeks
            env._state.cash -= severance
            env._state.costs += severance
            # Unassign from project/role
            if c.assigned_project and c.assigned_role:
                project = env._find_project(c.assigned_project)
                if project:
                    role = next((r for r in project.roles if r.role_id == c.assigned_role), None)
                    if role and candidate_id in role.assigned:
                        role.assigned.remove(candidate_id)
                        role.filled_count = max(0, role.filled_count - 1)
                        project.update_fill_status()
            del env._candidates[candidate_id]
            return {
                "success": True,
                "candidate_id": candidate_id,
                "severance_paid": round(severance, 2),
            }

        @mcp.tool()
        def request_project_extension(project_id: str) -> dict:
            """
            Ask client for a deadline extension on an expiring project.
            Grants 1–3 extra weeks but reduces client satisfaction.
            """
            project = env._find_project(project_id)
            if not project:
                return {"success": False, "error": f"Project {project_id} not found"}
            extension = env._rng.randint(1, 3)
            project.deadline_remaining += extension
            client = env._find_client(project.client_id)
            if client:
                event = {"type": "extension_requested", "project_id": project_id}
                result = env._llm.client_satisfaction(client, event, client.event_history)
                client.satisfaction_score = result.new_score
                client.churn_risk = result.churn_risk
                client.event_history.append(event)
            return {
                "success": True,
                "project_id": project_id,
                "extension_weeks": extension,
                "new_deadline_remaining": project.deadline_remaining,
                "client_satisfaction": round(client.satisfaction_score, 3) if client else None,
            }

        @mcp.tool()
        def pass_on_project(project_id: str) -> dict:
            """
            Decline a project you cannot fill. Avoids expiry penalty but loses revenue opportunity.
            """
            project = env._find_project(project_id)
            if not project:
                return {"success": False, "error": f"Project {project_id} not found"}
            client = env._find_client(project.client_id)
            if client:
                client.projects = [p for p in client.projects if p.project_id != project_id]
            return {"success": True, "project_id": project_id, "passed": True}

        # ── ADMIN / OVERRIDE TOOLS (for UI state manipulation) ─────────

        @mcp.tool()
        def _override_cash(amount: float) -> dict:
            """[Admin] Directly set cash balance. For UI/debug use only."""
            old = env._state.cash
            env._state.cash = float(amount)
            return {"success": True, "old_cash": round(old, 2), "new_cash": round(env._state.cash, 2)}

        @mcp.tool()
        def _override_satisfaction(client_id: str, score: float) -> dict:
            """[Admin] Directly set a client's satisfaction score. For UI/debug use only."""
            client = env._find_client(client_id)
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

    def _find_client(self, client_id: str) -> Client | None:
        return next((c for c in self._clients if c.client_id == client_id), None)

    def _find_project(self, project_id: str) -> Project | None:
        for client in self._clients:
            for p in client.projects:
                if p.project_id == project_id:
                    return p
        return None

    def _find_in_market(self, candidate_id: str) -> Candidate | None:
        return next((c for c in self._market if c.id == candidate_id), None)

    def _count_open_roles(self) -> int:
        return sum(
            (r.headcount - r.filled_count)
            for cl in self._clients
            for p in cl.projects
            for r in p.roles
            if not r.is_filled
        )

    def _snapshot(self) -> dict:
        """Compact state snapshot for observation metadata."""
        return {
            "step": self._state.step_count,
            "cash": round(self._state.cash, 2),
            "profit": round(self._state.revenue - self._state.costs, 2),
            "num_clients": len(self._clients),
            "num_candidates_on_payroll": len(self._candidates),
            "num_in_market": len(self._market),
            "avg_client_satisfaction": round(
                sum(cl.satisfaction_score for cl in self._clients) / len(self._clients)
                if self._clients else 0.0, 3
            ),
        }
