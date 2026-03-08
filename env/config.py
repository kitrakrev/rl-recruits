"""
All tunable parameters for the Staffing Agency RL environment.
Override via environment variables or pass a Config object to StaffingEnv.
"""
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # --- Episode ---
    episode_steps: int = 52          # 1 simulated business year
    seed_capital: float = 50_000.0
    reward_scale: float = 1_000.0    # divide raw $ reward by this for stability
    target_profit: float = 200_000.0
    win_bonus: float = 50_000.0

    # --- Curriculum stage ---
    curriculum_stage: Literal[1, 2, 3] = 3
    #   Stage 1: 1 client, 1 dev_type, 1-role projects
    #   Stage 2: 3 clients, 3 dev_types, 2-role projects
    #   Stage 3: full env

    # --- Clients ---
    num_clients: int = 3
    max_open_projects_per_client: int = 3
    project_arrival_lambda: float = 0.5   # Poisson λ per client per step
    churn_threshold: float = 0.3
    initial_satisfaction: float = 0.75
    client_ltv_estimate: float = 50_000.0  # used for churn penalty

    # --- Candidates ---
    market_pool_size: int = 20         # max candidates available in market
    t_patience: int = 8                # steps before unplaced candidate leaves
    contract_duration: int = 26        # weeks per placement contract

    # --- Projects ---
    t_deadline_min: int = 4
    t_deadline_max: int = 10
    max_roles_per_project: int = 3
    max_headcount_per_role: int = 2

    # --- Economics (from spec) ---
    onboarding_cost: float = 2_000.0
    severance_weeks: int = 2
    margin_pct: float = 0.25           # legacy, client_rate was salary × 1.25
    cost_per_interview: float = 500.0


    # --- LLM ---
    llm_mode: Literal["stub", "live"] = "stub"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # --- Legacy Rating tiers (to be removed once core.py is updated) ---
    rating_tiers: list = field(default_factory=list)

    # --- Developer type adjacency ---
    adjacency: dict = field(default_factory=lambda: {
        "backend":     {"backend", "fullstack", "ml_engineer"},
        "frontend":    {"frontend", "fullstack"},
        "fullstack":   {"fullstack", "backend", "frontend"},
        "ml_engineer": {"ml_engineer", "backend"},
        "devops":      {"devops"},
    })

    developer_types: list = field(default_factory=lambda: [
        "backend", "frontend", "fullstack", "ml_engineer", "devops"
    ])
    base_salaries: dict = field(default_factory=lambda: {
        "junior": 75_000, "mid": 110_000, "senior": 150_000
    })
    role_multipliers: dict = field(default_factory=lambda: {
        "frontend": 1.0, "backend": 1.05, "fullstack": 1.1, "ml_engineer": 1.3, "devops": 1.15
    })
    seniority_levels: list = field(default_factory=lambda: [
        "junior", "mid", "senior"
    ])

    industries: list = field(default_factory=lambda: [
        "fintech", "healthtech", "ecommerce", "saas", "logistics"
    ])

    def salary_from_rating(self, composite: float) -> tuple[float, float]:
        """Return (weekly_salary, weekly_client_rate) for a composite rating."""
        composite = max(1.0, min(5.0, composite))
        for lo, hi, _, annual_salary, annual_client in self.rating_tiers:
            if lo <= composite < hi:
                return annual_salary / 52, annual_client / 52
        # fallback to highest tier
        _, _, _, annual_salary, annual_client = self.rating_tiers[-1]
        return annual_salary / 52, annual_client / 52
