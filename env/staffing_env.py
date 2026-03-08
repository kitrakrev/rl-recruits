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
from .core import StaffingCore
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
        self.core = StaffingCore(self.config, self.llm, env_type="rl")

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
        self.core.reset(seed)
        self._expired_projects = []

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

        self.core.step_count += 1

        # Calculate current profit
        current_profit = self.core.revenue - self.core.costs
        
        # Check if profit target is hit
        profit_reached = current_profit >= self.config.target_profit

        # Apply a massive completion bonus so the agent WANTS to finish early
        if profit_reached:
            reward += self.config.win_bonus

        # Update termination logic to include the profit win condition
        terminated = (
            self.core.cash < 0 or 
            self.core.step_count >= self.config.episode_steps or 
            profit_reached
        )

        truncated = False

        obs = self._build_obs()
        info = self._build_info()
        info["tool_result"] = tool_result

        return obs, reward / self.config.reward_scale, terminated, truncated, info

    # ------------------------------------------------------------------
    # World tick
    # ------------------------------------------------------------------

    def _world_tick(self) -> float:
        reward = self.core.world_tick()
        # Ensure we sync expired list
        self._expired_projects = self.core.expired_projects[:]
        return reward

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool: str, params: dict) -> dict:
        handlers = {
            "get_agency_state":       self.core.tool_get_agency_state,
            "get_client_state":       self.core.tool_get_client_state,
            "get_candidate_state":    self.core.tool_get_candidate_state,
            "get_project_details":    self.core.tool_get_project_details,
            "get_candidate_profile":  self.core.tool_get_candidate_profile,
            "get_market_demand":      self.core.tool_get_market_demand,
            "get_financial_summary":  self.core.tool_get_financial_summary,
            "find_available_projects":      self.core.tool_find_available_projects,
            "confirm_project":              self.core.tool_confirm_project,
            "find_candidate":               self.core.tool_find_candidate,
            "interview_candidate":          self.core.tool_interview_candidate,
            "hire_candidate":               self.core.tool_hire_candidate,
            "negotiate_salary":             self.core.tool_negotiate_salary,
            "match_candidate_to_project":   self.core.tool_match_candidate_to_project,
            "let_go_candidate":             self.core.tool_let_go_candidate,
            "request_project_extension":    self.core.tool_request_project_extension,
            "pass_on_project":              self.core.tool_pass_on_project,
        }
        handler = handlers.get(tool)
        if handler is None:
            return {"success": False, "error": f"Unknown tool: {tool}", "reward": 0.0}
        return handler(**params)

    # ------------------------------------------------------------------
    # Observation vector (25-dim)
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        hired = [c for c in self.core.candidates.values() if c.status in ("hired", "placed")]
        placed = [c for c in self.core.candidates.values() if c.status == "placed"]
        benched = [c for c in self.core.candidates.values() if c.status == "hired"]
        pipeline = [c for c in self.core.candidates.values() if c.status == "in_pipeline"]
        burn = sum(c.salary_weekly for c in hired)
        runway = (self.core.cash / burn) if burn > 0 else 52.0
        placement_rate = len(placed) / len(hired) if hired else 0.0
        avg_ttp = float(np.mean(self.core.time_to_place)) if self.core.time_to_place else 0.0

        # Market demand (5-dim)
        demand = {dt: 0 for dt in self.config.developer_types}
        for client in self.core.clients:
            for p in client.projects:
                for role in p.roles:
                    if not role.is_filled:
                        demand[role.developer_type] += (role.headcount - role.filled_count)
        market_demand_vec = [float(demand[dt]) for dt in self.config.developer_types]

        # Project stats
        all_open = [p for cl in self.core.clients for p in cl.projects if p.fill_status != "SEALED"]
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
        active_clients = [cl for cl in self.core.clients if not cl.churn_risk]
        avg_sat = float(np.mean([cl.satisfaction_score for cl in active_clients])) if active_clients else 0.0
        num_at_churn_risk = sum(1 for cl in self.core.clients if cl.satisfaction_score < 0.4)

        # Candidate urgency
        churn_risk_count = sum(1 for c in self.core.candidates.values() if c.patience_remaining <= 2)
        avg_bench_weeks = float(np.mean([c.weeks_on_bench for c in benched])) if benched else 0.0

        obs = np.array([
            # Agency financials (4)
            self.core.cash / 1e4,
            (self.core.revenue - self.core.costs) / 1e4,
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
            "step": self.core.step_count,
            "cash": round(self.core.cash, 2),
            "profit": round(self.core.revenue - self.core.costs, 2),
            "num_clients": len(self.core.clients),
            "num_candidates": len(self.core.candidates),
            "num_market": len(self.core.market),
            "expired_projects_total": len(self._expired_projects),
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        agency = self.core.tool_get_agency_state()["state"]
        print(f"\\n=== Step {self.core.step_count}/{self.config.episode_steps} ===")
        print(f"  Cash:       ${agency['cash_balance']:>10,.0f}")
        print(f"  Profit:     ${agency['current_profit']:>10,.0f}")
        print(f"  Burn rate:  ${agency['burn_rate']:>10,.0f}/wk")
        print(f"  Runway:     {agency['cash_runway_weeks']:.1f} wks")
        print(f"  Hired:      {agency['num_candidates_hired']}  "
              f"Placed: {agency['num_candidates_placed']}  "
              f"Benched: {agency['num_candidates_benched']}")
        for cl in self.core.clients:
            print(f"  Client {cl.client_id}: sat={cl.satisfaction_score:.2f}  "
                  f"projects={len(cl.projects)}  churn={'YES' if cl.churn_risk else 'no'}")
