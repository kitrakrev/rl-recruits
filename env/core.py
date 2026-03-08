from __future__ import annotations
import asyncio
import random
import uuid
from typing import TYPE_CHECKING
import numpy as np

from .models import Candidate, Role, Project, Client, AgencyState
from .simulation import (
    generate_client, replenish_market, tick_project_arrivals,
    tick_project_deadlines, tick_contracts, tick_candidate_patience,
    compute_match_score, async_tick_project_deadlines,
    async_tick_candidate_patience
)

if TYPE_CHECKING:
    from .config import Config
    from .llm import LLMRouter


class StaffingCore:
    """
    Centralised staffing agency logic.
    Maintains the state and implements all the tool actions & world ticks.
    Used by both the RL environment and the MCP server.
    """
    def __init__(self, config: "Config", llm: "LLMRouter", env_type: str = "rl"):
        self.config = config
        self.llm = llm
        
        # Determine early termination triggers
        self.env_type = env_type

        # Runtime state (populated on reset)
        self.rng: random.Random = random.Random()
        self.step_count: int = 0
        self.cash: float = config.seed_capital
        self.revenue: float = 0.0
        self.costs: float = 0.0
        self.cumulative_reward: float = 0.0
        self.done: bool = False

        self.clients: list[Client] = []
        self.candidates: dict[str, Candidate] = {}
        self.market: list[Candidate] = []
        
        self.time_to_place: list[float] = []
        self.hire_step: dict[str, int] = {}
        self.pending_payments: list[float] = []
        self.expired_projects: list = []

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.step_count = 0
        self.cash = self.config.seed_capital
        self.revenue = 0.0
        self.costs = 0.0
        self.cumulative_reward = 0.0
        self.done = False

        self.candidates = {}
        self.market = []
        self.time_to_place = []
        self.hire_step = {}
        self.pending_payments = []
        self.expired_projects = []

        n_clients = {1: 1, 2: 3, 3: self.config.num_clients}.get(
            self.config.curriculum_stage, self.config.num_clients
        )
        self.clients = [
            generate_client(i, self.config, self.rng) for i in range(n_clients)
        ]

        replenish_market(self.market, self.config, self.rng)
        # Force initial projects to exist at step 0 so agent has work
        tick_project_arrivals(self.clients, self.config, self.rng)

    def world_tick(self) -> float:
        """Run step logic: billing, bench costs, project arrivals, deadlines."""
        reward = 0.0

        # 1. Billing: placed candidates generate margin
        for c in self.candidates.values():
            if c.status == "placed":
                self.cash += c.client_rate_weekly
                self.revenue += c.client_rate_weekly
                self.costs += c.salary_weekly
                self.cash -= c.salary_weekly
                reward += c.margin_weekly

        # 2. Bench burn
        for c in self.candidates.values():
            if c.status == "hired":
                self.cash -= c.salary_weekly
                self.costs += c.salary_weekly
                reward -= c.salary_weekly

        # 3. Project arrivals
        tick_project_arrivals(self.clients, self.config, self.rng)

        # 4. Project deadlines + expiry penalty
        expired = tick_project_deadlines(self.clients, self.llm)
        for project, client in expired:
            for role in project.roles:
                unfilled = role.headcount - role.filled_count
                if unfilled > 0:
                    penalty = unfilled * (85_000 / 52) * 1.25
                    reward -= penalty
                    self.costs += penalty
            # Keep consistent format for expired projects (store dicts)
            self.expired_projects.append(project.to_dict())

        # 5. Contract completions
        returning = tick_contracts(list(self.candidates.values()))
        for c in returning:
            c.status = "hired"
            c.assigned_project = None
            c.assigned_role = None
            c.contract_weeks_left = None
            c.weeks_on_bench = 0
            for client in self.clients:
                for project in client.projects:
                    for role in project.roles:
                        if c.id in role.assigned:
                            role.assigned.remove(c.id)
                            role.filled_count = max(0, role.filled_count - 1)
                            project.update_fill_status()

        # 6. Candidate patience
        agency_ctx = {
            "step": self.step_count,
            "open_roles": sum(r.headcount - r.filled_count for cl in self.clients for p in cl.projects for r in p.roles if not r.is_filled),
            "num_open_roles": sum(r.headcount - r.filled_count for cl in self.clients for p in cl.projects for r in p.roles if not r.is_filled),
        }
        leavers = tick_candidate_patience(
            list(self.candidates.values()), self.llm, agency_ctx
        )
        for c in leavers:
            self.candidates.pop(c.id, None)

        # 7. Client churn penalty
        for client in self.clients:
            if client.satisfaction_score < self.config.churn_threshold and not client.churn_risk:
                client.churn_risk = True
                reward -= self.config.client_ltv_estimate
                self.costs += self.config.client_ltv_estimate

        # 8. Replenish market
        replenish_market(self.market, self.config, self.rng)

        return reward

    async def async_world_tick(self) -> float:
        """Run step logic: billing, bench costs, project arrivals, deadlines."""
        reward = 0.0

        # 1. Billing: placed candidates generate margin
        for c in self.candidates.values():
            if c.status == "placed":
                self.cash += c.client_rate_weekly
                self.revenue += c.client_rate_weekly
                self.costs += c.salary_weekly
                self.cash -= c.salary_weekly
                reward += c.margin_weekly

        # 2. Bench burn
        for c in self.candidates.values():
            if c.status == "hired":
                self.cash -= c.salary_weekly
                self.costs += c.salary_weekly
                reward -= c.salary_weekly

        # 3. Project arrivals
        tick_project_arrivals(self.clients, self.config, self.rng)

        # 4. Project deadlines + expiry penalty
        expired = await async_tick_project_deadlines(self.clients, self.llm)
        for project, client in expired:
            for role in project.roles:
                unfilled = role.headcount - role.filled_count
                if unfilled > 0:
                    penalty = unfilled * (85_000 / 52) * 1.25
                    reward -= penalty
                    self.costs += penalty
            # Keep consistent format for expired projects (store dicts)
            self.expired_projects.append(project.to_dict())

        # 5. Contract completions
        returning = tick_contracts(list(self.candidates.values()))
        for c in returning:
            c.status = "hired"
            c.assigned_project = None
            c.assigned_role = None
            c.contract_weeks_left = None
            c.weeks_on_bench = 0
            for client in self.clients:
                for project in client.projects:
                    for role in project.roles:
                        if c.id in role.assigned:
                            role.assigned.remove(c.id)
                            role.filled_count = max(0, role.filled_count - 1)
                            project.update_fill_status()

        # 6. Candidate patience
        agency_ctx = {
            "step": self.step_count,
            "open_roles": sum(r.headcount - r.filled_count for cl in self.clients for p in cl.projects for r in p.roles if not r.is_filled),
            "num_open_roles": sum(r.headcount - r.filled_count for cl in self.clients for p in cl.projects for r in p.roles if not r.is_filled),
        }
        leavers = await async_tick_candidate_patience(
            list(self.candidates.values()), self.llm, agency_ctx
        )
        for c in leavers:
            self.candidates.pop(c.id, None)

        # 7. Client churn penalty
        for client in self.clients:
            if client.satisfaction_score < self.config.churn_threshold and not client.churn_risk:
                client.churn_risk = True
                reward -= self.config.client_ltv_estimate
                self.costs += self.config.client_ltv_estimate

        # 8. Replenish market
        replenish_market(self.market, self.config, self.rng)

        return reward

    # --- Tool methods return dict structure: {"success": bool, "reward": float, ...} ---

    def tool_get_agency_state(self) -> dict:
        hired = [c for c in self.candidates.values() if c.status in ("hired", "placed")]
        placed = [c for c in self.candidates.values() if c.status == "placed"]
        benched = [c for c in self.candidates.values() if c.status == "hired"]
        pipeline = [c for c in self.candidates.values() if c.status == "in_pipeline"]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self.cash / burn) if burn > 0 else 999.0
        placement_rate = len(placed) / len(hired) if hired else 0.0
        avg_ttp = float(np.mean(self.time_to_place)) if self.time_to_place else 0.0
        
        # Format for pure dictionaries (MCP) or AgencyState object
        state_dict = {
            "cash_balance": round(self.cash, 2),
            "current_revenue": round(self.revenue, 2),
            "current_costs": round(self.costs, 2),
            "current_profit": round(self.revenue - self.costs, 2),
            "num_candidates_hired": len(hired),
            "num_candidates_placed": len(placed),
            "num_candidates_benched": len(benched),
            "num_candidates_in_interview": len(pipeline),
            "placement_rate": round(placement_rate, 3),
            "avg_time_to_place": round(avg_ttp, 2),
            "pending_payments": self.pending_payments[:],
            "burn_rate": round(burn, 2),
            "cash_runway_weeks": round(min(runway, 999.0), 2),
            "step": self.step_count,
            "episode_steps": self.config.episode_steps,
        }
        return {"success": True, "reward": 0.0, "state": state_dict}


    async def async_tool_get_agency_state(self) -> dict:
        return self.tool_get_agency_state()
    def tool_get_client_state(self, client_id: str = "") -> dict:
        if client_id:
            c = next((cl for cl in self.clients if cl.client_id == client_id), None)
            if not c:
                return {"success": False, "error": f"Client {client_id} not found", "reward": 0.0}
            return {"success": True, "reward": 0.0, "state": c.to_dict()}
        return {"success": True, "reward": 0.0, "clients": [c.to_dict() for c in self.clients], "state": [c.to_dict() for c in self.clients]}


    async def async_tool_get_client_state(self, client_id: str = "") -> dict:
        return self.tool_get_client_state(client_id)
    def tool_get_candidate_state(self) -> dict:
        cands = list(self.candidates.values())
        avg_skill = float(np.mean([c.skill_score for c in cands])) if cands else 0.0
        avg_salary = float(np.mean([c.salary_weekly for c in cands if c.status in ("hired","placed")])) if cands else 0.0
        churn_flags = [c.id for c in cands if c.patience_remaining <= 2]
        return {
            "success": True, "reward": 0.0,
            "state": {  # Included for RL env format
                "num_candidates_available": len(self.market),
                "num_candidates_in_pipeline": sum(1 for c in cands if c.status == "in_pipeline"),
                "num_candidates_pending_hire": sum(1 for c in cands if c.status == "pending_hire"),
                "candidates_pool": [c.to_dict() for c in cands],
                "avg_skill_score": round(avg_skill, 3),
                "avg_salary_cost": round(avg_salary, 2),
                "churn_risk_flags": churn_flags,
            },
            # Flat attributes for MCP format
            "num_in_market": len(self.market),
            "num_in_pipeline": sum(1 for c in cands if c.status == "in_pipeline"),
            "num_pending_hire": sum(1 for c in cands if c.status == "pending_hire"),
            "num_hired": sum(1 for c in cands if c.status in ("hired", "placed")),
            "num_placed": sum(1 for c in cands if c.status == "placed"),
            "churn_risk_ids": churn_flags,
            "candidates": [c.to_dict() for c in cands],
        }


    async def async_tool_get_candidate_state(self) -> dict:
        return self.tool_get_candidate_state()
    def tool_get_project_details(self, project_id: str) -> dict:
        p = self.find_project(project_id)
        if not p:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        d = p.to_dict()
        return {"success": True, "reward": 0.0, "project": d, **d}


    async def async_tool_get_project_details(self, project_id: str) -> dict:
        return self.tool_get_project_details(project_id)
    def tool_get_candidate_profile(self, candidate_id: str) -> dict:
        c = self.candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not found", "reward": 0.0}
        d = c.to_dict()
        return {"success": True, "reward": 0.0, "candidate": d, **d}


    async def async_tool_get_candidate_profile(self, candidate_id: str) -> dict:
        return self.tool_get_candidate_profile(candidate_id)
    def tool_get_market_demand(self) -> dict:
        demand = {dt: 0 for dt in self.config.developer_types}
        for client in self.clients:
            if client.churn_risk:
                continue
            for project in client.projects:
                for role in project.roles:
                    if not role.is_filled:
                        demand[role.developer_type] += (role.headcount - role.filled_count)
        return {
            "success": True, "reward": 0.0,
            "demand": demand,  # RL
            "demand_by_type": demand, "total_open_slots": sum(demand.values()) # MCP
        }


    async def async_tool_get_market_demand(self) -> dict:
        return self.tool_get_market_demand()
    def tool_get_financial_summary(self) -> dict:
        hired = [c for c in self.candidates.values() if c.status in ("hired", "placed")]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self.cash / burn) if burn > 0 else 999.0
        summary = {
            "cash_balance": round(self.cash, 2),
            "revenue": round(self.revenue, 2),
            "costs": round(self.costs, 2),
            "profit": round(self.revenue - self.costs, 2),
            "burn_rate": round(burn, 2),
            "cash_runway_weeks": round(min(runway, 999.0), 2),
        }
        return {"success": True, "reward": 0.0, "summary": summary, **summary}


    async def async_tool_get_financial_summary(self) -> dict:
        return self.tool_get_financial_summary()
    def tool_find_available_projects(self) -> dict:
        projects = []
        for client in self.clients:
            if client.churn_risk:
                continue
            for p in client.projects:
                if p.fill_status != "SEALED":
                    d = p.to_dict()
                    d["client_industry"] = client.industry
                    d["client_satisfaction"] = client.satisfaction_score
                    projects.append(d)
        return {"success": True, "reward": 0.0, "projects": projects, "count": len(projects)}


    async def async_tool_find_available_projects(self) -> dict:
        return self.tool_find_available_projects()
    def tool_confirm_project(self, project_id: str) -> dict:
        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        project.confirmed = True
        client = next((c for c in self.clients if c.client_id == project.client_id), None)
        cli_score = None
        if client:
            event = {"type": "project_confirmed", "project_id": project_id}
            result = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
            cli_score = round(result.new_score, 3)
        return {"success": True, "reward": 0.0, "confirmed": True, "project_id": project_id, "client_satisfaction": cli_score}


    async def async_tool_confirm_project(self, project_id: str) -> dict:
        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        project.confirmed = True
        client = next((c for c in self.clients if c.client_id == project.client_id), None)
        cli_score = None
        if client:
            event = {"type": "project_confirmed", "project_id": project_id}
            result = await self.llm.async_client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
            cli_score = round(result.new_score, 3)
        return {"success": True, "reward": 0.0, "confirmed": True, "project_id": project_id, "client_satisfaction": cli_score}

    def tool_find_candidate(self, developer_type: str = "") -> dict:
        if developer_type:
            found = [c for c in self.market if c.developer_type == developer_type]
        else:
            found = self.market[:]
        return {
            "success": True, "reward": 0.0,
            "candidates": [c.to_dict() for c in found],
            "count": len(found),
        }


    async def async_tool_find_candidate(self, developer_type: str = "") -> dict:
        return self.tool_find_candidate(developer_type)
    def tool_interview_candidate(self, candidate_id: str) -> dict:
        c = next((m for m in self.market if m.id == candidate_id), None)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not in market", "reward": 0.0}
        
        job_desc = f"{c.developer_type} {c.seniority_level}"
        if c.seniority_level != "junior":
           job_desc += " role"
        result = self.llm.interview(c, job_desc)
        c.base_rating = result.base_rating
        c.technical_score = result.technical_score
        c.communication = result.communication
        c.culture_fit = result.culture_fit
        c.red_flags = result.red_flags
        c.interview_summary = result.summary
        c.status = "in_pipeline"
        
        self.market.remove(c)
        self.candidates[c.id] = c
        
        return {
            "success": True, "reward": 0.0,
            "candidate_id": c.id,
            "base_rating": result.base_rating,
            "proceed": result.proceed,
            "summary": result.summary,
            "red_flags": result.red_flags,
            "technical_score": result.technical_score,
            "result": {  # legacy RL wrapper
                "base_rating": result.base_rating,
                "proceed": result.proceed,
                "summary": result.summary,
                "red_flags": result.red_flags,
            }
        }


    async def async_tool_interview_candidate(self, candidate_id: str) -> dict:
        c = next((m for m in self.market if m.id == candidate_id), None)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not in market", "reward": 0.0}
        
        job_desc = f"{c.developer_type} {c.seniority_level}"
        if c.seniority_level != "junior":
           job_desc += " role"
        result = await self.llm.async_interview(c, job_desc)
        c.base_rating = result.base_rating
        c.technical_score = result.technical_score
        c.communication = result.communication
        c.culture_fit = result.culture_fit
        c.red_flags = result.red_flags
        c.interview_summary = result.summary
        c.status = "in_pipeline"
        
        self.market.remove(c)
        self.candidates[c.id] = c
        
        return {
            "success": True, "reward": 0.0,
            "candidate_id": c.id,
            "base_rating": result.base_rating,
            "proceed": result.proceed,
            "summary": result.summary,
            "red_flags": result.red_flags,
            "technical_score": result.technical_score,
            "result": {  # legacy RL wrapper
                "base_rating": result.base_rating,
                "proceed": result.proceed,
                "summary": result.summary,
                "red_flags": result.red_flags,
            }
        }

    def tool_hire_candidate(self, candidate_id: str) -> dict:
        c = self.candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not found", "reward": 0.0}
        if c.status not in ("in_pipeline", "pending_hire"):
            return {"success": False, "error": f"Cannot hire: status is '{c.status}'", "reward": 0.0}

        self.cash -= self.config.onboarding_cost
        self.costs += self.config.onboarding_cost
        reward = -self.config.onboarding_cost

        composite = round(0.4 * c.base_rating + 0.6 * 3.0, 2)
        c.composite_rating = composite
        c.salary_weekly, c.client_rate_weekly = self.config.salary_from_rating(composite)
        c.margin_weekly = c.client_rate_weekly - c.salary_weekly
        c.status = "hired"
        self.hire_step[c.id] = self.step_count

        return {
            "success": True, "reward": reward, "hired": True,
            "candidate_id": c.id,
            "developer_type": c.developer_type,
            "seniority": c.seniority_level,
            "composite_rating": composite,
            "salary_weekly": round(c.salary_weekly, 2),
            "onboarding_cost": self.config.onboarding_cost,
            "bench_burn_per_week": round(c.salary_weekly, 2),
            "break_even_weeks": round(self.config.onboarding_cost / c.margin_weekly, 1) if c.margin_weekly else 999.0,
        }


    async def async_tool_hire_candidate(self, candidate_id: str) -> dict:
        return self.tool_hire_candidate(candidate_id)
    def tool_negotiate_salary(self, candidate_id: str, offer_weekly: float) -> dict:
        c = self.candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not found", "reward": 0.0}
        market_ctx = {"step": self.step_count, "market_size": len(self.market), "market_pool_size": len(self.market)}
        result = self.llm.salary_negotiation(c, offer_weekly, market_ctx)
        c.patience_remaining = max(0, c.patience_remaining + result.patience_impact)
        if result.accepted:
            c.salary_expectation = offer_weekly
            c.status = "pending_hire"
        return {
            "success": True, "reward": 0.0,
            "accepted": result.accepted,
            "counter_offer": result.counter_offer,
            "reason": result.acceptance_reason,
            "candidate_expectation": c.salary_expectation,
        }


    async def async_tool_negotiate_salary(self, candidate_id: str, offer_weekly: float) -> dict:
        c = self.candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not found", "reward": 0.0}
        market_ctx = {"step": self.step_count, "market_size": len(self.market), "market_pool_size": len(self.market)}
        result = await self.llm.async_salary_negotiation(c, offer_weekly, market_ctx)
        c.patience_remaining = max(0, c.patience_remaining + result.patience_impact)
        if result.accepted:
            c.salary_expectation = offer_weekly
            c.status = "pending_hire"
        return {
            "success": True, "reward": 0.0,
            "accepted": result.accepted,
            "counter_offer": result.counter_offer,
            "reason": result.acceptance_reason,
            "candidate_expectation": c.salary_expectation,
        }

    def tool_match_candidate_to_project(self, candidate_id: str, project_id: str, role_id: str) -> dict:
        c = self.candidates.get(candidate_id)
        if not c or c.status != "hired":
            return {"success": False, "error": "Candidate must be in 'hired' status / available", "reward": 0.0}

        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}

        role = next((r for r in project.roles if r.role_id == role_id), None)
        if not role:
            return {"success": False, "error": f"Role {role_id} not found", "reward": 0.0}

        if role.is_filled:
            return {"success": False, "error": "Role is already fully filled", "reward": 0.0}

        match_score = compute_match_score(c, role, self.config)
        if match_score == 0.0:
            return {
                "success": False, 
                "error": "Illegal assignment \u2014 type mismatch or insufficient skill" if self.env_type=="mcp" else "Illegal assignment \u2014 skill or type mismatch", 
                "reward": 0.0,
                "candidate_type": c.developer_type, "role_type": role.developer_type,
                "candidate_skill": c.skill_score, "role_min_skill": role.min_skill_score
            }

        client = next((cl for cl in self.clients if cl.client_id == project.client_id), None)
        project_ctx = {
            "client_industry": client.industry if client else "unknown",
            "project_id": project_id,
        }
        fit_result = self.llm.project_fit(c, role, project_ctx)

        c.project_fit_rating = fit_result.project_fit_rating
        c.composite_rating = fit_result.composite_rating
        c.salary_weekly, c.client_rate_weekly = self.config.salary_from_rating(c.composite_rating)
        c.margin_weekly = c.client_rate_weekly - c.salary_weekly
        c.status = "placed"
        c.assigned_project = project_id
        c.assigned_role = role_id
        c.contract_weeks_left = self.config.contract_duration
        c.weeks_on_bench = 0

        role.filled_count += 1
        role.assigned.append(candidate_id)
        project.update_fill_status()

        sealed_now = project.fill_status == "SEALED"
        reward = 0.0

        if client:
            evt_type = "project_sealed" if sealed_now else ("adjacent_match" if match_score < 1.0 else "partial_fill")
            event = {"type": evt_type, "project_id": project_id, "match_score": match_score}
            
            if self.env_type == "rl":
                 event["fit_rating"] = fit_result.project_fit_rating
                 
            sat = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = sat.new_score
            client.churn_risk = sat.churn_risk
            client.event_history.append(event)
            if sealed_now:
                client.num_projects_filled += 1
                total_hc = sum(r.headcount for r in project.roles)
                project.weekly_revenue = client.contracted_rate * total_hc

        if sealed_now and project.confirmed:
            original_deadline = self.config.t_deadline_max
            weeks_taken = original_deadline - project.deadline_remaining
            if weeks_taken <= 2:
                speed_bonus = c.margin_weekly * 0.1
                reward += speed_bonus if self.env_type == "rl" else speed_bonus / self.config.reward_scale

        if c.id in self.hire_step:
            ttp = self.step_count - self.hire_step[c.id]
            self.time_to_place.append(float(ttp))

        return {
            "success": True, "reward": reward,
            "match_score": match_score,
            "composite_rating": c.composite_rating,
            "project_fit_rating": fit_result.project_fit_rating,
            "fit_rationale": fit_result.fit_rationale,
            "project_sealed": sealed_now,
            "weekly_margin": round(c.margin_weekly, 2),
            "speed_bonus": round(reward * self.config.reward_scale if self.env_type == "mcp" else reward, 2),
            "risk_flags": getattr(fit_result, "risk_flags", []),
        }


    async def async_tool_match_candidate_to_project(self, candidate_id: str, project_id: str, role_id: str) -> dict:
        c = self.candidates.get(candidate_id)
        if not c or c.status != "hired":
            return {"success": False, "error": "Candidate must be in 'hired' status / available", "reward": 0.0}

        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}

        role = next((r for r in project.roles if r.role_id == role_id), None)
        if not role:
            return {"success": False, "error": f"Role {role_id} not found", "reward": 0.0}

        if role.is_filled:
            return {"success": False, "error": "Role is already fully filled", "reward": 0.0}

        match_score = compute_match_score(c, role, self.config)
        if match_score == 0.0:
            return {
                "success": False, 
                "error": "Illegal assignment \u2014 type mismatch or insufficient skill" if self.env_type=="mcp" else "Illegal assignment \u2014 skill or type mismatch", 
                "reward": 0.0,
                "candidate_type": c.developer_type, "role_type": role.developer_type,
                "candidate_skill": c.skill_score, "role_min_skill": role.min_skill_score
            }

        client = next((cl for cl in self.clients if cl.client_id == project.client_id), None)
        project_ctx = {
            "client_industry": client.industry if client else "unknown",
            "project_id": project_id,
        }
        fit_result = await self.llm.async_project_fit(c, role, project_ctx)

        c.project_fit_rating = fit_result.project_fit_rating
        c.composite_rating = fit_result.composite_rating
        c.salary_weekly, c.client_rate_weekly = self.config.salary_from_rating(c.composite_rating)
        c.margin_weekly = c.client_rate_weekly - c.salary_weekly
        c.status = "placed"
        c.assigned_project = project_id
        c.assigned_role = role_id
        c.contract_weeks_left = self.config.contract_duration
        c.weeks_on_bench = 0

        role.filled_count += 1
        role.assigned.append(candidate_id)
        project.update_fill_status()

        sealed_now = project.fill_status == "SEALED"
        reward = 0.0

        if client:
            evt_type = "project_sealed" if sealed_now else ("adjacent_match" if match_score < 1.0 else "partial_fill")
            event = {"type": evt_type, "project_id": project_id, "match_score": match_score}
            
            if self.env_type == "rl":
                 event["fit_rating"] = fit_result.project_fit_rating
                 
            sat = await self.llm.async_client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = sat.new_score
            client.churn_risk = sat.churn_risk
            client.event_history.append(event)
            if sealed_now:
                client.num_projects_filled += 1
                total_hc = sum(r.headcount for r in project.roles)
                project.weekly_revenue = client.contracted_rate * total_hc

        if sealed_now and project.confirmed:
            original_deadline = self.config.t_deadline_max
            weeks_taken = original_deadline - project.deadline_remaining
            if weeks_taken <= 2:
                speed_bonus = c.margin_weekly * 0.1
                reward += speed_bonus if self.env_type == "rl" else speed_bonus / self.config.reward_scale

        if c.id in self.hire_step:
            ttp = self.step_count - self.hire_step[c.id]
            self.time_to_place.append(float(ttp))

        return {
            "success": True, "reward": reward,
            "match_score": match_score,
            "composite_rating": c.composite_rating,
            "project_fit_rating": fit_result.project_fit_rating,
            "fit_rationale": fit_result.fit_rationale,
            "project_sealed": sealed_now,
            "weekly_margin": round(c.margin_weekly, 2),
            "speed_bonus": round(reward * self.config.reward_scale if self.env_type == "mcp" else reward, 2),
            "risk_flags": getattr(fit_result, "risk_flags", []),
        }

    def tool_let_go_candidate(self, candidate_id: str) -> dict:
        c = self.candidates.get(candidate_id)
        if not c:
            return {"success": False, "error": f"Candidate {candidate_id} not found", "reward": 0.0}
        severance = c.salary_weekly * self.config.severance_weeks
        self.cash -= severance
        self.costs += severance
        reward = -severance
        if c.assigned_project and c.assigned_role:
            project = self.find_project(c.assigned_project)
            if project:
                role = next((r for r in project.roles if r.role_id == c.assigned_role), None)
                if role and candidate_id in role.assigned:
                    role.assigned.remove(candidate_id)
                    role.filled_count = max(0, role.filled_count - 1)
                    project.update_fill_status()
        self.candidates.pop(candidate_id, None)
        return {"success": True, "reward": reward, "severance_paid": round(severance, 2), "candidate_id": candidate_id}


    async def async_tool_let_go_candidate(self, candidate_id: str) -> dict:
        return self.tool_let_go_candidate(candidate_id)
    def tool_request_project_extension(self, project_id: str) -> dict:
        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        extension = self.rng.randint(1, 3)
        project.deadline_remaining += extension
        client = next((cl for cl in self.clients if cl.client_id == project.client_id), None)
        cli_score = None
        if client:
            event = {"type": "extension_requested", "project_id": project_id}
            result = self.llm.client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
            cli_score = round(client.satisfaction_score, 3)
        return {
            "success": True, "reward": 0.0,
            "project_id": project_id,
            "extension_weeks": extension,
            "new_deadline": project.deadline_remaining,
            "new_deadline_remaining": project.deadline_remaining,
            "client_satisfaction": cli_score,
        }


    async def async_tool_request_project_extension(self, project_id: str) -> dict:
        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        extension = self.rng.randint(1, 3)
        project.deadline_remaining += extension
        client = next((cl for cl in self.clients if cl.client_id == project.client_id), None)
        cli_score = None
        if client:
            event = {"type": "extension_requested", "project_id": project_id}
            result = await self.llm.async_client_satisfaction(client, event, client.event_history)
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
            cli_score = round(client.satisfaction_score, 3)
        return {
            "success": True, "reward": 0.0,
            "project_id": project_id,
            "extension_weeks": extension,
            "new_deadline": project.deadline_remaining,
            "new_deadline_remaining": project.deadline_remaining,
            "client_satisfaction": cli_score,
        }

    def tool_pass_on_project(self, project_id: str) -> dict:
        project = self.find_project(project_id)
        if not project:
            return {"success": False, "error": f"Project {project_id} not found", "reward": 0.0}
        client = next((cl for cl in self.clients if cl.client_id == project.client_id), None)
        if client:
            client.projects = [p for p in client.projects if p.project_id != project_id]
        return {"success": True, "reward": 0.0, "project_id": project_id, "passed": True}

    # --- Helpers ---

    async def async_tool_pass_on_project(self, project_id: str) -> dict:
        return self.tool_pass_on_project(project_id)
    def find_project(self, project_id: str) -> Project | None:
        for cl in self.clients:
            for p in cl.projects:
                if p.project_id == project_id:
                    return p
        return None
