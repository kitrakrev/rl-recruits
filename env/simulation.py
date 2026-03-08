from __future__ import annotations
import asyncio
"""
Simulation logic — stochastic world dynamics that run each step.
Handles: project arrivals, deadlines, candidate patience, contracts, churn.
"""
import random
import uuid
from typing import TYPE_CHECKING

from .models import Candidate, Role, Project, Client

if TYPE_CHECKING:
    from .config import Config
    from .llm import LLMRouter


_counters: dict[str, int] = {}

def _reset_counters() -> None:
    """Call at env reset so each episode gets fresh sequential IDs starting from 1."""
    _counters.clear()

def _next_id(prefix: str) -> str:
    n = _counters.get(prefix, 0) + 1
    _counters[prefix] = n
    return f"{prefix}{n}"

def _uid() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidate(config: "Config", rng: random.Random) -> Candidate:
    dev_type = rng.choice(config.developer_types)
    seniority = rng.choices(
        config.seniority_levels, weights=[0.40, 0.40, 0.20]
    )[0]
    # skill_score: Beta(2,2) rescaled to [0.3, 1.0]
    raw = rng.betavariate(2, 2)
    skill = round(0.3 + raw * 0.7, 3)

    # Dynamic salary expectation: base × role_multiplier × skill_modifier × ±20% rng variance
    base = config.base_salaries.get(seniority, 85_000)
    mult = config.role_multipliers.get(dev_type, 1.0)

    # skill_score biases expectation: high-skill candidates demand more
    skill_modifier = 0.8 + skill * 0.4   # maps [0.3,1.0] → [0.92, 1.20]

    # ±20% pure RNG variance on top
    variance = rng.uniform(0.80, 1.20)
    annual_expectation = base * mult * skill_modifier * variance

    expectation_weekly = round(annual_expectation / 52, 2)

    patience = config.t_patience + rng.randint(-2, 2)
    patience = max(4, patience)

    cid = _next_id("C")
    return Candidate(
        id=cid,
        developer_type=dev_type,
        seniority_level=seniority,
        skill_score=skill,
        salary_expectation=expectation_weekly,
        patience_remaining=patience,
        ttl_weeks=patience,   # organic market TTL mirrors patience budget
        status="available",
    )



# ---------------------------------------------------------------------------
# Project / Role generation
# ---------------------------------------------------------------------------

def generate_project(client: Client, config: "Config", rng: random.Random) -> Project:
    pid = _next_id("P")
    deadline = rng.randint(config.t_deadline_min, config.t_deadline_max)
    n_roles = rng.randint(1, config.max_roles_per_project)

    # In curriculum stage 1, restrict to 1 role
    if config.curriculum_stage == 1:
        n_roles = 1

    roles = []
    for i in range(n_roles):
        rid = f"R{pid[1:]}-{i}"  # e.g. P3 → R3-0, R3-1
        dev_type = rng.choice(config.developer_types)
        if config.curriculum_stage == 1:
            dev_type = config.developer_types[0]
        seniority = rng.choices(
            config.seniority_levels, weights=[0.35, 0.45, 0.20]
        )[0]
        min_skill = round(rng.uniform(0.3, 0.8), 2)
        headcount = rng.randint(1, config.max_headcount_per_role)

        # Variable bill rate: what the client will pay for this role.
        # Range: 1.3× to 2.0× the base salary depending on role scarcity & market.
        # This is the MAXIMUM revenue the agency can earn — margin = bill_rate - hired_salary.
        base = config.base_salaries.get(seniority, 85_000)
        mult = config.role_multipliers.get(dev_type, 1.0)
        # Scarcity premium: devops/ml_engineer commands higher bill rates
        scarcity = {"ml_engineer": 1.15, "devops": 1.10, "backend": 1.05,
                    "fullstack": 1.02, "frontend": 1.0}.get(dev_type, 1.0)
        # Client willingness-to-pay: uniform draw [1.3×, 2.0×] of base×role×scarcity
        client_premium = rng.uniform(1.30, 2.00)
        bill_rate_annual = base * mult * scarcity * client_premium

        roles.append(Role(
            role_id=rid,
            developer_type=dev_type,
            seniority=seniority,
            min_skill_score=min_skill,
            headcount=headcount,
            bill_rate_weekly=round(bill_rate_annual / 52, 2)
        ))


    return Project(
        project_id=pid,
        client_id=client.client_id,
        roles=roles,
        deadline_remaining=deadline,
    )


def generate_client(client_idx: int, config: "Config", rng: random.Random) -> Client:
    industry = rng.choice(config.industries)
    cid = f"CL{client_idx + 1}"
    return Client(
        client_id=cid,
        industry=industry,
        satisfaction_score=config.initial_satisfaction,
    )


# ---------------------------------------------------------------------------
# Step-level world dynamics
# ---------------------------------------------------------------------------

def tick_project_arrivals(
    clients: list[Client],
    config: "Config",
    rng: random.Random,
) -> list[Project]:
    """Stochastically add new projects for each client (Poisson arrivals)."""
    new_projects = []
    for client in clients:
        if client.churn_risk:
            continue  # churned clients stop submitting
        open_count = client.num_open_projects
        if open_count >= config.max_open_projects_per_client:
            continue
        # Poisson: P(k arrivals) where λ = config.project_arrival_lambda
        arrivals = _poisson_sample(config.project_arrival_lambda, rng)
        for _ in range(arrivals):
            if client.num_open_projects >= config.max_open_projects_per_client:
                break
            project = generate_project(client, config, rng)
            client.projects.append(project)
            client.num_projects_submitted += 1
            new_projects.append(project)
    return new_projects


def _poisson_sample(lam: float, rng: random.Random) -> int:
    """Simple Poisson sampler via Knuth algorithm."""
    import math
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def tick_project_deadlines(
    clients: list[Client],
    llm: "LLMRouter",
) -> list[tuple[Project, Client]]:
    """Decrement deadlines; return list of (expired_project, client) pairs."""
    expired = []
    for client in clients:
        still_active = []
        for project in client.projects:
            if project.fill_status == "SEALED":
                still_active.append(project)
                continue
            project.deadline_remaining -= 1
            if project.deadline_remaining <= 0:
                expired.append((project, client))
                client.num_projects_expired += 1
                # Update satisfaction
                event = {"type": "project_expired", "project_id": project.project_id}
                result = llm.client_satisfaction(client, event, client.event_history)
                client.satisfaction_score = result.new_score
                client.churn_risk = result.churn_risk
                client.event_history.append(event)
            else:
                still_active.append(project)
        client.projects = still_active
    return expired

async def async_tick_project_deadlines(
    clients: list[Client],
    llm: "LLMRouter",
) -> list[tuple[Project, Client]]:

    expired = []
    tasks = []
    for client in clients:
        still_active = []
        for project in client.projects:
            if project.fill_status == "SEALED":
                still_active.append(project)
                continue
            project.deadline_remaining -= 1
            if project.deadline_remaining <= 0:
                expired.append((project, client))
                client.num_projects_expired += 1
                # Prepare LLM call
                event = {"type": "project_expired", "project_id": project.project_id}
                tasks.append((client, event, llm.async_client_satisfaction(client, event, client.event_history)))
            else:
                still_active.append(project)
        client.projects = still_active

    if tasks:
        import asyncio
        results = await asyncio.gather(*(t[2] for t in tasks))
        for (client, event, _), result in zip(tasks, results):
            client.satisfaction_score = result.new_score
            client.churn_risk = result.churn_risk
            client.event_history.append(event)
    return expired


def tick_candidate_patience(
    candidates: list[Candidate],
    llm: "LLMRouter",
    agency_context: dict,
) -> list[Candidate]:
    """Decrement patience for benched candidates; return list of leavers."""
    leavers = []
    for c in candidates:
        if c.status not in ("hired",):
            continue  # only benched (hired but not placed) candidates age
        c.weeks_on_bench += 1
        c.patience_remaining -= 1

        if c.patience_remaining <= 2:
            result = llm.candidate_leave(c, agency_context)
            if result.leave:
                leavers.append(c)
    return leavers

async def async_tick_candidate_patience(
    candidates: list[Candidate],
    llm: "LLMRouter",
    agency_context: dict,
) -> list[Candidate]:
    leavers = []
    tasks = []
    
    for c in candidates:
        if c.status not in ("hired",):
            continue
        c.weeks_on_bench += 1
        c.patience_remaining -= 1

        if c.patience_remaining <= 2:
            tasks.append((c, llm.async_candidate_leave(c, agency_context)))

    if tasks:
        results = await asyncio.gather(*(t[1] for t in tasks))
        for (c, _), result in zip(tasks, results):
            if result.leave:
                leavers.append(c)

    return leavers


def tick_contracts(
    candidates: list[Candidate],
) -> list[Candidate]:
    """Decrement contract weeks; return candidates whose contracts completed."""
    returning = []
    for c in candidates:
        if c.status != "placed" or c.contract_weeks_left is None:
            continue
        c.contract_weeks_left -= 1
        if c.contract_weeks_left <= 0:
            returning.append(c)
    return returning


def tick_market_churn(market: list[Candidate], rng: random.Random) -> None:
    """Organic churn: market candidates age out if not interviewed.

    Each week uninterviewed, patience_remaining ticks down.  When it hits 0 the
    candidate leaves — simulating real-world passivity (they get another offer,
    lose interest, etc.).  This prevents the market from becoming a stale pool of
    indefinitely available candidates, forcing the agent to act with urgency.
    """
    to_remove = []
    for c in market:
        c.patience_remaining -= 1
        if c.patience_remaining <= 0:
            to_remove.append(c)
    for c in to_remove:
        market.remove(c)


def replenish_market(
    market: list[Candidate],
    config: "Config",
    rng: random.Random,
) -> None:
    """Keep market pool up to max size, after ticking churn."""
    tick_market_churn(market, rng)
    while len(market) < config.market_pool_size:
        market.append(generate_candidate(config, rng))



# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

_SKILL_TOLERANCE = 0.20  # allow up to 20% below min_skill_score

def compute_match_score(
    candidate: Candidate,
    role: Role,
    config: "Config",
) -> float:
    """Return match score 0.0–1.0; 0.0 means hard-blocked (only on type mismatch)."""
    adjacent_types = config.adjacency.get(candidate.developer_type, set())

    # Only hard-block on type — all other mismatches reduce score but don't block
    if role.developer_type not in adjacent_types:
        return 0.0

    # Skill below floor (after 20% tolerance) → soft penalty, not a block
    effective_min = role.min_skill_score * (1 - _SKILL_TOLERANCE)
    skill_ok = candidate.skill_score >= effective_min
    skill_penalty = 0.0 if skill_ok else 0.15  # score reduction if below tolerance

    # Seniority mismatch reduces score but no longer blocks
    seniority_penalty = 0.0 if _seniority_ok(candidate.seniority_level, role.seniority) else 0.15

    exact_type      = candidate.developer_type == role.developer_type
    exact_seniority = candidate.seniority_level == role.seniority

    if exact_type and exact_seniority:
        base = 1.0
    elif not exact_type:
        base = 0.7   # adjacent type
    else:
        base = 0.85  # seniority overqualified

    return max(0.3, base - skill_penalty - seniority_penalty)


def diagnose_match_failure(
    candidate: Candidate,
    role: Role,
    config: "Config",
) -> str:
    """Return a precise failure message. Only called when compute_match_score == 0 (type block)."""
    adjacent_types = config.adjacency.get(candidate.developer_type, set())
    if role.developer_type not in adjacent_types:
        return (
            f"TYPE MISMATCH: candidate '{candidate.id}' is type='{candidate.developer_type}'. "
            f"Role '{role.role_id}' requires type='{role.developer_type}' (or adjacent). "
            f"Valid types for this candidate: {sorted(adjacent_types)}. "
            f"Find a role that needs '{candidate.developer_type}', or hire a '{role.developer_type}' candidate."
        )
    # Should not reach here since only type=0 triggers hard block now
    return f"Match blocked for candidate '{candidate.id}' on role '{role.role_id}'."


def _seniority_ok(candidate_sen: str, role_sen: str) -> bool:
    """Senior can fill any role; mid fills junior/mid; junior fills junior only."""
    order = {"junior": 0, "mid": 1, "senior": 2}
    return order.get(candidate_sen, 0) >= order.get(role_sen, 0)
