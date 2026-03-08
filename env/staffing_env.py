"""
StaffingEnv — main OpenEnv-compatible RL environment.

Interface:
    env = StaffingEnv(config)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

Action format (dict):
    {"tool": "hire_candidate", "params": {"candidate_id": "C-BA-abc12345"}}

Observation: flat float32 numpy array (25-dim, see Section 10 of spec).
"""
from __future__ import annotations
import random
import uuid
from typing import Any

import numpy as np

from .config import Config
from .models import Candidate, Role, Project, Client, AgencyState
from .llm import LLMRouter
from .simulation import (
    generate_client,
    generate_candidate,
    replenish_market,
    tick_project_arrivals,
    tick_project_deadlines,
    tick_candidate_patience,
    tick_contracts,
    compute_match_score,
)


class StaffingEnv:
    """
    Single-agent RL environment: Staffing Agency maximises profit over 52 steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.llm = LLMRouter(self.config)

        # These are set during reset()
        self._rng: random.Random = random.Random()
        self._step: int = 0
        self._cash: float = 0.0
        self._revenue: float = 0.0
        self._costs: float = 0.0

        self._clients: list[Client] = []
        self._candidates: dict[str, Candidate] = {}  # id → Candidate
        self._market: list[Candidate] = []            # available but not yet interviewed
        self._pending_payments: list[float] = []

        self._time_to_place: list[float] = []         # for rolling avg
        self._hire_step: dict[str, int] = {}          # candidate_id → step when hired

        # Expired projects this episode (for info)
        self._expired_projects: list[dict] = []

    # ------------------------------------------------------------------
    # Spaces (OpenEnv-compatible)
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> dict:
        return {
            "shape": (25,),
            "dtype": "float32",
            "low": -np.inf,
            "high": np.inf,
        }

    @property
    def action_space(self) -> dict:
        return {
            "type": "dict",
            "tools": [
                "get_agency_state", "get_client_state", "get_candidate_state",
                "get_project_details", "get_candidate_profile",
                "get_market_demand", "get_financial_summary",
                "find_available_projects", "confirm_project",
                "find_candidate", "interview_candidate", "hire_candidate",
                "negotiate_salary", "match_candidate_to_project",
                "let_go_candidate", "request_project_extension", "pass_on_project",
            ],
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._rng = random.Random(seed)
        self._step = 0
        self._cash = self.config.seed_capital
        self._revenue = 0.0
        self._costs = 0.0
        self._candidates = {}
        self._market = []
        self._pending_payments = []
        self._time_to_place = []
        self._hire_step = {}
        self._expired_projects = []

        # Generate clients
        n_clients = {1: 1, 2: 3, 3: self.config.num_clients}[self.config.curriculum_stage]
        self._clients = [
            generate_client(i, self.config, self._rng) for i in range(n_clients)
        ]

        # Populate market
        replenish_market(self._market, self.config, self._rng)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not isinstance(action, dict) or "tool" not in action:
            raise ValueError("Action must be dict with 'tool' key. Got: " + str(action))

        reward = 0.0
        tool = action["tool"]
        params = action.get("params", {})

        # --- Execute agent action ---
        tool_result = self._dispatch(tool, params)
        reward += tool_result.get("reward", 0.0)

        # --- World tick (run once per step regardless of action) ---
        step_reward = self._world_tick()
        reward += step_reward

        self._step += 1

        terminated = self._cash < 0 or self._step >= self.config.episode_steps
        truncated = False

        obs = self._build_obs()
        info = self._build_info()
        info["tool_result"] = tool_result

        return obs, reward / self.config.reward_scale, terminated, truncated, info

    # ------------------------------------------------------------------
    # World tick
    # ------------------------------------------------------------------

    def _world_tick(self) -> float:
        reward = 0.0

        # 1. Billing: revenue from placed candidates
        for c in self._candidates.values():
            if c.status == "placed":
                self._cash += c.client_rate_weekly
                self._revenue += c.client_rate_weekly
                self._costs += c.salary_weekly
                self._cash -= c.salary_weekly
                reward += c.margin_weekly

        # 2. Bench costs: salary for benched (hired but not placed) candidates
        for c in self._candidates.values():
            if c.status == "hired":
                self._cash -= c.salary_weekly
                self._costs += c.salary_weekly
                reward -= c.salary_weekly

        # 3. Project arrivals
        tick_project_arrivals(self._clients, self.config, self._rng)

        # 4. Project deadline countdown + expiry penalty
        expired = tick_project_deadlines(self._clients, self.llm)
        for project, client in expired:
            # Opportunity cost penalty
            for role in project.roles:
                unfilled = role.headcount - role.filled_count
                if unfilled > 0:
                    # Use average weekly client rate as penalty proxy
                    penalty = unfilled * (85_000 / 52) * 1.25
                    reward -= penalty
                    self._costs += penalty
            self._expired_projects.append(project.to_dict())

        # 5. Contract completions → return to bench
        all_candidates = list(self._candidates.values())
        returning = tick_contracts(all_candidates)
        for c in returning:
            c.status = "hired"
            c.assigned_project = None
            c.assigned_role = None
            c.contract_weeks_left = None
            c.weeks_on_bench = 0
            # Update project role assignment
            for client in self._clients:
                for project in client.projects:
                    for role in project.roles:
                        if c.id in role.assigned:
                            role.assigned.remove(c.id)
                            role.filled_count = max(0, role.filled_count - 1)
                            project.update_fill_status()

        # 6. Candidate patience decay + voluntary leavers
        agency_ctx = {"step": self._step, "num_open_roles": self._count_open_roles()}
        leavers = tick_candidate_patience(all_candidates, self.llm, agency_ctx)
        for c in leavers:
            del self._candidates[c.id]

        # 7. Client churn check
        for client in self._clients:
            if client.satisfaction_score < self.config.churn_threshold and not client.churn_risk:
                client.churn_risk = True
                reward -= self.config.client_ltv_estimate
                self._costs += self.config.client_ltv_estimate

        # 8. Replenish market
        replenish_market(self._market, self.config, self._rng)

        return reward

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool: str, params: dict) -> dict:
        handlers = {
            # GET tools
            "get_agency_state":       self._tool_get_agency_state,
            "get_client_state":       self._tool_get_client_state,
            "get_candidate_state":    self._tool_get_candidate_state,
            "get_project_details":    self._tool_get_project_details,
            "get_candidate_profile":  self._tool_get_candidate_profile,
            "get_market_demand":      self._tool_get_market_demand,
            "get_financial_summary":  self._tool_get_financial_summary,
            # EXECUTE tools
            "find_available_projects":      self._tool_find_available_projects,
            "confirm_project":              self._tool_confirm_project,
            "find_candidate":               self._tool_find_candidate,
            "interview_candidate":          self._tool_interview_candidate,
            "hire_candidate":               self._tool_hire_candidate,
            "negotiate_salary":             self._tool_negotiate_salary,
            "match_candidate_to_project":   self._tool_match_candidate_to_project,
            "let_go_candidate":             self._tool_let_go_candidate,
            "request_project_extension":    self._tool_request_project_extension,
            "pass_on_project":              self._tool_pass_on_project,
        }
        handler = handlers.get(tool)
        if handler is None:
            return {"success": False, "error": f"Unknown tool: {tool}", "reward": 0.0}
        return handler(**params)

    # ------------------------------------------------------------------
    # GET tools
    # ------------------------------------------------------------------

    def _tool_get_agency_state(self) -> dict:
        hired = [c for c in self._candidates.values() if c.status in ("hired", "placed")]
        placed = [c for c in self._candidates.values() if c.status == "placed"]
        benched = [c for c in self._candidates.values() if c.status == "hired"]
        pipeline = [c for c in self._candidates.values() if c.status == "in_pipeline"]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self._cash / burn) if burn > 0 else float("inf")
        placement_rate = len(placed) / len(hired) if hired else 0.0
        avg_ttp = float(np.mean(self._time_to_place)) if self._time_to_place else 0.0
        state = AgencyState(
            cash_balance=round(self._cash, 2),
            current_revenue=round(self._revenue, 2),
            current_costs=round(self._costs, 2),
            current_profit=round(self._revenue - self._costs, 2),
            num_candidates_hired=len(hired),
            num_candidates_placed=len(placed),
            num_candidates_benched=len(benched),
            num_candidates_in_interview=len(pipeline),
            placement_rate=round(placement_rate, 3),
            avg_time_to_place=round(avg_ttp, 2),
            pending_payments=self._pending_payments[:],
            burn_rate=round(burn, 2),
            cash_runway_weeks=round(runway, 2),
        )
        return {"success": True, "reward": 0.0, "state": state.to_dict()}

    def _tool_get_client_state(self, client_id: str | None = None) -> dict:
        if client_id:
            client = self._find_client(client_id)
            if not client:
                return {"success": False, "error": "Client not found", "reward": 0.0}
            return {"success": True, "reward": 0.0, "state": client.to_dict()}
        return {
            "success": True, "reward": 0.0,
            "state": [c.to_dict() for c in self._clients],
        }

    def _tool_get_candidate_state(self) -> dict:
        pool = [c.to_dict() for c in self._candidates.values()]
        avg_skill = float(np.mean([c.skill_score for c in self._candidates.values()])) if self._candidates else 0.0
        avg_salary = float(np.mean([c.salary_weekly for c in self._candidates.values() if c.status in ("hired","placed")])) if self._candidates else 0.0
        churn_flags = [c.id for c in self._candidates.values() if c.patience_remaining <= 2]
        return {
            "success": True, "reward": 0.0,
            "state": {
                "num_candidates_available": len(self._market),
                "num_candidates_in_pipeline": sum(1 for c in self._candidates.values() if c.status == "in_pipeline"),
                "num_candidates_pending_hire": sum(1 for c in self._candidates.values() if c.status == "pending_hire"),
                "candidates_pool": pool,
                "avg_skill_score": round(avg_skill, 3),
                "avg_salary_cost": round(avg_salary, 2),
                "churn_risk_flags": churn_flags,
            },
        }

    def _tool_get_project_details(self, project_id: str) -> dict:
        project = self._find_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found", "reward": 0.0}
        return {"success": True, "reward": 0.0, "project": project.to_dict()}

    def _tool_get_candidate_profile(self, candidate_id: str) -> dict:
        c = self._candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": "Candidate not found", "reward": 0.0}
        return {"success": True, "reward": 0.0, "candidate": c.to_dict()}

    def _tool_get_market_demand(self) -> dict:
        demand = {dt: 0 for dt in self.config.developer_types}
        for client in self._clients:
            if client.churn_risk:
                continue
            for project in client.projects:
                for role in project.roles:
                    if not role.is_filled:
                        demand[role.developer_type] = demand.get(role.developer_type, 0) + (role.headcount - role.filled_count)
        return {"success": True, "reward": 0.0, "demand": demand}

    def _tool_get_financial_summary(self) -> dict:
        hired = [c for c in self._candidates.values() if c.status in ("hired", "placed")]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self._cash / burn) if burn > 0 else float("inf")
        return {
            "success": True, "reward": 0.0,
            "summary": {
                "cash_balance": round(self._cash, 2),
                "revenue": round(self._revenue, 2),
                "costs": round(self._costs, 2),
                "profit": round(self._revenue - self._costs, 2),
                "burn_rate": round(burn, 2),
                "cash_runway_weeks": round(runway, 2),
            },
        }

    # ------------------------------------------------------------------
    # EXECUTE tools
    # ------------------------------------------------------------------

    def _tool_find_available_projects(self) -> dict:
        projects = []
        for client in self._clients:
            if client.churn_risk:
                continue
            for p in client.projects:
                if p.fill_status != "SEALED":
                    projects.append(p.to_dict())
        return {"success": True, "reward": 0.0, "projects": projects}

    def _tool_confirm_project(self, project_id: str) -> dict:
        project = self._find_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found", "reward": 0.0}
        project.confirmed = True
        client = self._find_client(project.client_id)
        if client:
            event = {"type": "project_confirmed", "project_id": project_id}
            result = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
        return {"success": True, "reward": 0.0, "confirmed": True}

    def _tool_find_candidate(self, developer_type: str | None = None) -> dict:
        if developer_type:
            found = [c for c in self._market if c.developer_type == developer_type]
        else:
            found = self._market[:]
        return {"success": True, "reward": 0.0, "candidates": [c.to_dict() for c in found]}

    def _tool_interview_candidate(self, candidate_id: str) -> dict:
        c = self._find_in_market(candidate_id)
        if not c:
            return {"success": False, "error": "Candidate not in market", "reward": 0.0}
        job_desc = f"{c.developer_type} {c.seniority_level} role"
        result = self.llm.interview(c, job_desc)
        c.base_rating = result.base_rating
        c.technical_score = result.technical_score
        c.communication = result.communication
        c.culture_fit = result.culture_fit
        c.red_flags = result.red_flags
        c.interview_summary = result.summary
        c.status = "in_pipeline"
        # Move from market to tracked candidates
        self._market.remove(c)
        self._candidates[c.id] = c
        return {
            "success": True, "reward": 0.0,
            "result": {
                "base_rating": result.base_rating,
                "proceed": result.proceed,
                "summary": result.summary,
                "red_flags": result.red_flags,
            },
        }

    def _tool_hire_candidate(self, candidate_id: str) -> dict:
        c = self._candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": "Candidate not found", "reward": 0.0}
        if c.status not in ("in_pipeline", "pending_hire"):
            return {"success": False, "error": f"Cannot hire candidate in status: {c.status}", "reward": 0.0}

        # Deduct onboarding cost
        self._cash -= self.config.onboarding_cost
        self._costs += self.config.onboarding_cost
        reward = -self.config.onboarding_cost

        # Set salary based on composite (use base_rating × 0.4 + 3 × 0.6 as pre-placement composite)
        composite = round(0.4 * c.base_rating + 0.6 * 3.0, 2)
        c.composite_rating = composite
        c.salary_weekly, c.client_rate_weekly = self.config.salary_from_rating(composite)
        c.margin_weekly = c.client_rate_weekly - c.salary_weekly
        c.status = "hired"
        self._hire_step[c.id] = self._step

        return {
            "success": True,
            "reward": reward,
            "hired": True,
            "salary_weekly": c.salary_weekly,
            "onboarding_cost": self.config.onboarding_cost,
        }

    def _tool_negotiate_salary(self, candidate_id: str, offer: float) -> dict:
        c = self._candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": "Candidate not found", "reward": 0.0}
        market_ctx = {"step": self._step, "market_pool_size": len(self._market)}
        result = self.llm.salary_negotiation(c, offer, market_ctx)
        c.patience_remaining = max(0, c.patience_remaining + result.patience_impact)
        if result.accepted:
            c.salary_expectation = offer
            c.status = "pending_hire"
        return {
            "success": True, "reward": 0.0,
            "accepted": result.accepted,
            "counter_offer": result.counter_offer,
            "reason": result.acceptance_reason,
        }

    def _tool_match_candidate_to_project(
        self, candidate_id: str, project_id: str, role_id: str
    ) -> dict:
        c = self._candidates.get(candidate_id)
        if not c or c.status not in ("hired",):
            return {"success": False, "error": "Candidate not available for placement", "reward": 0.0}

        project = self._find_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found", "reward": 0.0}

        role = next((r for r in project.roles if r.role_id == role_id), None)
        if not role:
            return {"success": False, "error": "Role not found", "reward": 0.0}

        if role.is_filled:
            return {"success": False, "error": "Role already fully filled", "reward": 0.0}

        match_score = compute_match_score(c, role, self.config)
        if match_score == 0.0:
            return {"success": False, "error": "Illegal assignment — skill or type mismatch", "reward": 0.0}

        # LLM fit evaluation
        project_ctx = {"client_industry": self._find_client(project.client_id).industry if self._find_client(project.client_id) else "unknown"}
        fit_result = self.llm.project_fit(c, role, project_ctx)

        # Update candidate
        c.project_fit_rating = fit_result.project_fit_rating
        c.composite_rating = fit_result.composite_rating
        c.salary_weekly, c.client_rate_weekly = self.config.salary_from_rating(c.composite_rating)
        c.margin_weekly = c.client_rate_weekly - c.salary_weekly
        c.status = "placed"
        c.assigned_project = project_id
        c.assigned_role = role_id
        c.contract_weeks_left = self.config.contract_duration
        c.weeks_on_bench = 0

        # Update role
        role.filled_count += 1
        role.assigned.append(candidate_id)

        # Update project
        project.update_fill_status()

        reward = 0.0
        sealed_now = project.fill_status == "SEALED"

        # Client satisfaction update
        client = self._find_client(project.client_id)
        if client:
            event_type = "project_sealed" if sealed_now else "partial_fill"
            if match_score < 1.0:
                event_type = "adjacent_match"
            event = {
                "type": event_type,
                "project_id": project_id,
                "match_score": match_score,
                "fit_rating": fit_result.project_fit_rating,
            }
            sat_result = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = sat_result.new_score
            client.churn_risk = sat_result.churn_risk
            client.event_history.append(event)
            if sealed_now:
                client.num_projects_filled += 1
                # Set project weekly revenue
                total_headcount = sum(r.headcount for r in project.roles)
                project.weekly_revenue = client.contracted_rate * total_headcount

        # Speed bonus: sealed within 2 weeks of confirmation
        if sealed_now and project.confirmed:
            original_deadline = self.config.t_deadline_max
            weeks_taken = original_deadline - project.deadline_remaining
            if weeks_taken <= 2:
                speed_bonus = c.margin_weekly * 0.1
                reward += speed_bonus

        # Track time-to-place
        if c.id in self._hire_step:
            ttp = self._step - self._hire_step[c.id]
            self._time_to_place.append(ttp)

        return {
            "success": True,
            "reward": reward,
            "match_score": match_score,
            "composite_rating": c.composite_rating,
            "project_sealed": sealed_now,
            "fit_rationale": fit_result.fit_rationale,
        }

    def _tool_let_go_candidate(self, candidate_id: str) -> dict:
        c = self._candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": "Candidate not found", "reward": 0.0}

        severance = c.salary_weekly * self.config.severance_weeks
        self._cash -= severance
        self._costs += severance
        reward = -severance

        # Unassign from project/role if placed
        if c.assigned_project and c.assigned_role:
            project = self._find_project(c.assigned_project)
            if project:
                role = next((r for r in project.roles if r.role_id == c.assigned_role), None)
                if role and candidate_id in role.assigned:
                    role.assigned.remove(candidate_id)
                    role.filled_count = max(0, role.filled_count - 1)
                    project.update_fill_status()

        del self._candidates[candidate_id]
        return {
            "success": True,
            "reward": reward,
            "severance_paid": severance,
        }

    def _tool_request_project_extension(self, project_id: str) -> dict:
        project = self._find_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found", "reward": 0.0}
        extension = self._rng.randint(1, 3)
        project.deadline_remaining += extension
        client = self._find_client(project.client_id)
        if client:
            event = {"type": "extension_requested", "project_id": project_id}
            result = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
        return {
            "success": True, "reward": 0.0,
            "extension_weeks": extension,
            "new_deadline": project.deadline_remaining,
        }

    def _tool_pass_on_project(self, project_id: str) -> dict:
        project = self._find_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found", "reward": 0.0}
        client = self._find_client(project.client_id)
        if client:
            client.projects = [p for p in client.projects if p.project_id != project_id]
        return {"success": True, "reward": 0.0, "passed": True}

    # ------------------------------------------------------------------
    # Observation vector (25-dim)
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        hired = [c for c in self._candidates.values() if c.status in ("hired", "placed")]
        placed = [c for c in self._candidates.values() if c.status == "placed"]
        benched = [c for c in self._candidates.values() if c.status == "hired"]
        pipeline = [c for c in self._candidates.values() if c.status == "in_pipeline"]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self._cash / burn) if burn > 0 else 52.0
        placement_rate = len(placed) / len(hired) if hired else 0.0
        avg_ttp = float(np.mean(self._time_to_place)) if self._time_to_place else 0.0

        # Market demand (5-dim)
        demand = {dt: 0 for dt in self.config.developer_types}
        for client in self._clients:
            for p in client.projects:
                for role in p.roles:
                    if not role.is_filled:
                        demand[role.developer_type] += (role.headcount - role.filled_count)
        market_demand_vec = [float(demand[dt]) for dt in self.config.developer_types]

        # Project stats
        all_open = [p for cl in self._clients for p in cl.projects if p.fill_status != "SEALED"]
        num_projects_pending = len(all_open)
        total_open_slots = sum(
            (r.headcount - r.filled_count)
            for p in all_open for r in p.roles
        )
        # Deadline histogram [0-2wk, 3-5wk, 6+wk]
        hist = [0, 0, 0]
        for p in all_open:
            d = p.deadline_remaining
            if d <= 2:
                hist[0] += 1
            elif d <= 5:
                hist[1] += 1
            else:
                hist[2] += 1

        # Client health
        active_clients = [cl for cl in self._clients if not cl.churn_risk]
        avg_sat = float(np.mean([cl.satisfaction_score for cl in active_clients])) if active_clients else 0.0
        num_at_churn_risk = sum(1 for cl in self._clients if cl.satisfaction_score < 0.4)

        # Candidate urgency
        churn_risk_count = sum(1 for c in self._candidates.values() if c.patience_remaining <= 2)
        avg_bench_weeks = float(np.mean([c.weeks_on_bench for c in benched])) if benched else 0.0

        obs = np.array([
            # Agency financials (4)
            self._cash / 1e4,
            (self._revenue - self._costs) / 1e4,
            burn / 1e3,
            min(runway, 52.0) / 52.0,
            # Staffing metrics (6)
            float(len(hired)),
            float(len(placed)),
            float(len(benched)),
            float(len(pipeline)),
            placement_rate,
            avg_ttp,
            # Demand signals (5 + 1 + 1 + 3 = 10)
            *market_demand_vec,
            float(num_projects_pending),
            float(total_open_slots),
            *[float(h) for h in hist],
            # Client health (3)
            float(len(active_clients)),
            avg_sat,
            float(num_at_churn_risk),
            # Candidate urgency (2)
            float(churn_risk_count),
            avg_bench_weeks,
        ], dtype=np.float32)

        assert obs.shape == (25,), f"Obs shape mismatch: {obs.shape}"
        return obs

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self) -> dict:
        return {
            "step": self._step,
            "cash": round(self._cash, 2),
            "profit": round(self._revenue - self._costs, 2),
            "num_clients": len(self._clients),
            "num_candidates": len(self._candidates),
            "num_market": len(self._market),
            "expired_projects_total": len(self._expired_projects),
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        agency = self._tool_get_agency_state()["state"]
        print(f"\n=== Step {self._step}/{self.config.episode_steps} ===")
        print(f"  Cash:       ${agency['cash_balance']:>10,.0f}")
        print(f"  Profit:     ${agency['current_profit']:>10,.0f}")
        print(f"  Burn rate:  ${agency['burn_rate']:>10,.0f}/wk")
        print(f"  Runway:     {agency['cash_runway_weeks']:.1f} wks")
        print(f"  Hired:      {agency['num_candidates_hired']}  "
              f"Placed: {agency['num_candidates_placed']}  "
              f"Benched: {agency['num_candidates_benched']}")
        for cl in self._clients:
            print(f"  Client {cl.client_id}: sat={cl.satisfaction_score:.2f}  "
                  f"projects={len(cl.projects)}  churn={'YES' if cl.churn_risk else 'no'}")

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
