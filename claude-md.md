# CLAUDE.md — Staffing Agency Agentic RL Environment

> This file defines the complete design specification for the Staffing Agency OpenEnv —
> an agentic Reinforcement Learning environment where the agent maximises the profit
> of a staffing agency by recruiting candidates and placing them into client projects.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Actors](#2-actors)
3. [Assumptions & Economic Model](#3-assumptions--economic-model)
4. [State Space](#4-state-space)
5. [Action Space](#5-action-space)
6. [Tools](#6-tools)
7. [Project & Matching Model](#7-project--matching-model)
8. [Reward Function](#8-reward-function)
9. [Episode Structure](#9-episode-structure)
10. [Observation Vector](#10-observation-vector)
11. [Open Technical & Research Questions](#11-open-technical--research-questions)

---

## 1. Project Overview

A single-agent RL environment where the **Staffing Agency is the RL agent**. The agent
must optimally source, interview, hire, and place candidates into multi-role client
projects to maximise net profit.

**Primary Goal:** Maximise cumulative profit over a 52-step episode (1 simulated business year).

**Core Tension:**
- **Over-hiring** → benched candidates bleed salary with $0 revenue
- **Under-hiring** → client projects expire, satisfaction drops, clients churn
- **Optimal policy** lives in the narrow band between these two failure modes

---

## 2. Actors

| Actor | Role | Controlled By |
|---|---|---|
| **Staffing Agency** | The RL agent — recruits, places, and manages candidates | Agent (learned policy) |
| **Candidates** | Supply side — available to be recruited and placed | Environment (simulated) |
| **Clients** | Demand side — submit multi-role projects, pay on sealing | Environment (simulated) |

---

## 3. Assumptions & Economic Model

### Structural Assumptions
- Each project belongs to **one specific client** and contains **multiple roles**.
- Each role specifies a `developer_type`, `seniority_level`, `min_skill_score`, and `headcount`.
- A project is **SEALED** (active/revenue-generating) only when **all roles are filled**.
- Partial fills generate **$0 revenue** — the agent bears full salary burn while filling.
- A candidate can only be assigned to **one active role** at a time.
- Client projects arrive **stochastically** each step (Poisson-distributed per client).
- Candidates have a **patience window** — if not placed within `T_patience` steps, they leave.
- Client projects have a **deadline** — unfulfilled projects expire after `T_deadline` steps.
- Time is **discrete**: 1 step = 1 business week.

### Rating System

Every candidate receives two ratings:

- **Base Rating** — assessed during `interview_candidate()`. Reflects general skill, communication, and professionalism. Scale: **1–5** (integer).
- **Project Fit Rating** — assessed during `match_candidate_to_project()`. Reflects how well the candidate matches the specific role requirements (type, seniority, skill_score). Scale: **1–5** (integer).
- **Composite Rating** — the single value used to set salary and client income:

```
composite_rating = 0.4 × base_rating + 0.6 × project_fit_rating
```

> The composite rating is re-evaluated at each new placement. A candidate can have
> different composite ratings across different projects.

---

### Rating → Salary & Income Tiers

All compensation is **driven by the candidate's composite rating**. Neither salary nor client rate is fixed.

| Rating | Label | Candidate Salary (Annual) | Candidate Salary (Weekly) | Client Rate (Annual) | Client Rate (Weekly) | Agency Margin (Weekly) |
|---|---|---|---|---|---|---|
| 1.0 – 1.9 | ⭐ Poor | $55,000 | $1,058 | $68,750 | $1,322 | **+$265** |
| 2.0 – 2.9 | ⭐⭐ Below Average | $70,000 | $1,346 | $87,500 | $1,683 | **+$337** |
| 3.0 – 3.9 | ⭐⭐⭐ Average | $85,000 | $1,635 | $106,250 | $2,043 | **+$408** |
| 4.0 – 4.4 | ⭐⭐⭐⭐ Good | $105,000 | $2,019 | $131,250 | $2,524 | **+$504** |
| 4.5 – 5.0 | ⭐⭐⭐⭐⭐ Excellent | $130,000 | $2,500 | $162,500 | $3,125 | **+$625** |

> **Margin is always 25% of client rate** — the agency's fixed markup regardless of tier.
> `client_rate = candidate_salary × 1.25`

**Fixed costs (rating-independent):**

| Item | Cost |
|---|---|
| Onboarding (one-time, on hire) | −$2,000 |
| Severance (on let_go) | −2 weeks of candidate salary at current rating |
| Bench cost (per week, unplaced) | −candidate salary (weekly) at their rating |

---

### Economic Implications by Rating

| Scenario | Weekly P&L |
|---|---|
| Rating-5 candidate placed | +$625/wk margin |
| Rating-3 candidate placed | +$408/wk margin |
| Rating-1 candidate placed | +$265/wk margin |
| Rating-5 candidate benched | −$2,500/wk (highest burn risk) |
| Rating-1 candidate benched | −$1,058/wk (lower burn but also lower upside) |

> **Key insight for policy:** High-rated candidates are more profitable when placed but
> **catastrophically expensive when benched**. The agent must learn to pipeline high-rated
> candidates only when a matching role is imminent, and use lower-rated candidates as
> lower-risk bench fill for easier roles.

---

### Billing Model
- Revenue is **recurring weekly** for every candidate actively placed on a sealed project.
- Agency invoices client at `client_rate` (weekly) based on candidate's composite rating.
- Contract duration per placement: configurable (default 26 weeks / 6 months).
- When a contract ends, the candidate returns to benched status — salary continues.

---

## 4. State Space

All state is observed via **GET tools** (see Section 6). The environment supports
**full observability** by default (MDP); partial observability (POMDP) is a research option.

### 4.1 Agency State — `get_agency_state()`

| Attribute | Type | Description |
|---|---|---|
| `cash_balance` | float | Liquid cash available for decisions |
| `current_revenue` | float | Cumulative revenue this episode |
| `current_costs` | float | Cumulative costs (hiring, salary, severance) |
| `current_profit` | float | `revenue − costs` |
| `num_candidates_hired` | int | Total candidates currently on payroll |
| `num_candidates_placed` | int | Candidates actively placed on a sealed project |
| `num_candidates_benched` | int | Hired but unplaced — idle, accruing salary cost |
| `num_candidates_in_interview` | int | Candidates in the interview pipeline |
| `placement_rate` | float | `placed / hired` ratio — efficiency metric |
| `avg_time_to_place` | float | Rolling average steps from hire → placement |
| `pending_payments` | list[float] | Client payments in transit |
| `burn_rate` | float | Weekly salary outflow across all hired candidates |
| `cash_runway_weeks` | float | `cash_balance / burn_rate` — solvency horizon |

### 4.2 Client State — `get_client_state()` (per client)

| Attribute | Type | Description |
|---|---|---|
| `client_id` | str | Unique client identifier |
| `industry` | str | Sector (e.g. fintech, healthtech) — shapes developer_type demand mix |
| `contracted_rate` | float | Fixed $100,000/yr ($1,923/wk) per placed candidate |
| `billing_cycle` | str | `"weekly"` |
| `num_open_projects` | int | Projects with at least one unfilled role |
| `num_projects_submitted` | int | Lifetime projects submitted |
| `num_projects_filled` | int | Successfully sealed projects |
| `num_projects_expired` | int | Timed-out projects (penalty trigger) |
| `projects` | list[Project] | See Section 7 for Project schema |
| `satisfaction_score` | float | 0–1; drops on expiry/slow fills, rises on fast/perfect fills |
| `churn_risk` | bool | `True` if `satisfaction_score < churn_threshold` |
| `total_billed` | float | Cumulative amount invoiced to this client |
| `total_paid` | float | Cumulative amount received from this client |

### 4.3 Candidate State — `get_candidate_state()`

| Attribute | Type | Description |
|---|---|---|
| `num_candidates_available` | int | In the market, not yet interviewed |
| `num_candidates_in_pipeline` | int | Interviewed, awaiting hire decision |
| `num_candidates_pending_hire` | int | Offer extended, awaiting acceptance |
| `candidates_pool` | list[Candidate] | See Candidate schema below |
| `avg_skill_score` | float | Average skill level across hired candidates |
| `avg_salary_cost` | float | Average weekly salary across bench |
| `churn_risk_flags` | list[str] | Candidate IDs with `patience_remaining ≤ 2` |

**Candidate Schema:**
```
Candidate {
  id:                   str
  developer_type:       str       # "backend" | "ml_engineer" | "devops" | "frontend" | "fullstack"
  seniority_level:      str       # "junior" | "mid" | "senior"
  skill_score:          float     # 0.0 – 1.0
  base_rating:          int       # 1–5, set after interview_candidate()
  project_fit_rating:   int       # 1–5, set after match_candidate_to_project()
  composite_rating:     float     # 0.4×base + 0.6×fit; drives salary & client rate
  salary_weekly:        float     # derived from composite_rating tier
  client_rate_weekly:   float     # derived from composite_rating tier (salary × 1.25)
  margin_weekly:        float     # client_rate - salary = 25% of client_rate
  salary_expectation:   float     # candidate's minimum acceptable weekly salary (pre-hire)
  patience_remaining:   int       # steps before candidate leaves if not placed
  status:               str       # "available"|"in_pipeline"|"pending_hire"|"hired"|"placed"
  assigned_project:     str|null  # project_id if placed
  assigned_role:        str|null  # role_id if placed
  weeks_on_bench:       int       # steps hired but unplaced
  contract_weeks_left:  int|null  # remaining weeks on active placement
}
```

---

## 5. Action Space

Actions are executed via **EXECUTE tools** (see Section 6).
All actions are taken by the **Staffing Agency agent** each step.

### 5.1 Agent Actions (EXECUTE)

| Action | Parameters | Effect | Cost / Reward |
|---|---|---|---|
| `find_available_projects()` | — | Refresh project queue from client market | Neutral (observation refresh) |
| `confirm_project(project_id)` | `project_id` | Lock in a project, signalling commitment to client | Satisfaction signal |
| `find_candidate(developer_type)` | `developer_type` | Surface available candidates from market | Neutral |
| `interview_candidate(candidate_id)` | `candidate_id` | Move candidate to pipeline | Small time cost |
| `hire_candidate(candidate_id)` | `candidate_id` | Place on payroll | −$2,000 onboarding + −$1,538/wk salary |
| `negotiate_salary(candidate_id, offer)` | `candidate_id`, `float` | Adjust salary offer; affects acceptance probability | Variable |
| `match_candidate_to_project(candidate_id, project_id, role_id)` | all three IDs | Assign candidate to a specific role within a project. Revenue starts **only when all roles filled** | **+$1,923/wk when project sealed** |
| `let_go_candidate(candidate_id)` | `candidate_id` | Remove from payroll | −$3,076 severance |
| `request_project_extension(project_id)` | `project_id` | Ask client for deadline extension | −satisfaction score |
| `pass_on_project(project_id)` | `project_id` | Decline a project the agency cannot fill | Avoids expiry penalty; loses opportunity |

### 5.2 Environment-Simulated Actions

**Client (simulated):**

| Action | Trigger | Effect |
|---|---|---|
| `submit_project(roles[], deadline)` | Stochastic each step | Adds multi-role project to client's queue |
| `release_payment(project_id)` | On project SEALED | Adds $1,923/wk × headcount to agency cash |
| `expire_project(project_id)` | `deadline_remaining == 0` | Removes project; applies penalty to agency |
| `update_satisfaction(client_id)` | Each step | Adjusts score based on fill rate, speed, match quality |
| `churn_client(client_id)` | `satisfaction_score < threshold` | Client stops submitting projects |

**Candidate (simulated):**

| Action | Trigger | Effect |
|---|---|---|
| `accept_offer(candidate_id)` | After `hire_candidate` with adequate offer | Candidate joins payroll |
| `reject_offer(candidate_id)` | Poor offer or excessive wait | Candidate exits pipeline |
| `leave_agency(candidate_id)` | `patience_remaining == 0` | Candidate exits; bench shrinks |
| `complete_contract(candidate_id)` | `contract_weeks_left == 0` | Candidate returns to benched status |

---

## 6. Tools

Tools are split into two categories. The agent calls these as part of its decision loop.

### GET Tools — State Observation

```python
get_agency_state()                    # Full agency financial + placement snapshot
get_client_state(client_id=None)      # Per-client or all-client project/satisfaction state
get_candidate_state()                 # Pool, pipeline, bench, churn risk
get_project_details(project_id)       # Roles, deadline, client, match requirements
get_candidate_profile(candidate_id)   # Full skill, salary, patience, assignment details
get_market_demand()                   # Projected developer_type demand for next N steps
get_financial_summary()               # P&L, burn rate, cash runway
```

### EXECUTE Tools — Actions

```python
find_available_projects()                                   # Discovery
confirm_project(project_id)                                 # Commitment
find_candidate(developer_type)                              # Discovery
interview_candidate(candidate_id)                           # Pipeline
hire_candidate(candidate_id)                                # Hiring
negotiate_salary(candidate_id, offer)                       # Hiring
match_candidate_to_project(candidate_id, project_id,        # Placement (core action)
                            role_id)
let_go_candidate(candidate_id)                              # Bench management
request_project_extension(project_id)                       # Client management
pass_on_project(project_id)                                 # Risk management
```

---

## 7. Project & Matching Model

### Project Schema

```
Project {
  project_id:          str
  client_id:           str
  roles:               list[Role]
  deadline_remaining:  int            # steps until expiry
  fill_status:         str            # "OPEN" | "PARTIAL" | "SEALED"
  match_score:         float          # avg match quality across filled roles
  confirmed:           bool           # agent has called confirm_project()
  weekly_revenue:      float          # $0 until SEALED; $1,923 × total headcount when SEALED
}

Role {
  role_id:        str
  developer_type: str       # "backend" | "frontend" | "fullstack" | "ml_engineer" | "devops"
  seniority:      str       # "junior" | "mid" | "senior"
  min_skill_score: float    # minimum acceptable skill_score
  headcount:      int       # number of candidates needed for this role
  filled_count:   int       # candidates currently assigned
  assigned:       list[str] # candidate_ids
}
```

### Matching Logic — `match_candidate_to_project(candidate_id, project_id, role_id)`

A valid match requires all three conditions:

```
1. candidate.developer_type == role.developer_type   (or adjacent — see below)
2. candidate.skill_score >= role.min_skill_score
3. candidate.seniority_level compatible with role.seniority
```

**Match Score Outcomes:**

| Scenario | match_score | Effect |
|---|---|---|
| Perfect match (all 3 conditions met exactly) | 1.0 | Full satisfaction gain |
| Adjacent type (e.g. fullstack → frontend) | 0.7 | Partial satisfaction; client notified |
| Seniority mismatch only (overqualified) | 0.85 | Minor satisfaction dip |
| skill_score < min_skill_score | 0.0 | **Action blocked** — illegal assignment |
| Type mismatch | 0.0 | **Action blocked** — illegal assignment |

> Adjacent match types must be pre-defined in a **compatibility matrix** (e.g. `fullstack` is
> adjacent to `frontend` and `backend`; `ml_engineer` is adjacent to `backend`).

**Project Sealing:**
A project transitions to `SEALED` when `filled_count == headcount` for **every role**.
Only at this point does weekly revenue begin flowing from the client.

---

## 8. Reward Function

Computed at every time step `t`:

```
R(t) = R_billing(t)
     - C_salary(t)
     - C_onboarding(t)
     - C_severance(t)
     - C_expired(t)
     - C_churn(t)
     + B_speed(t)          # optional shaping bonus
```

| Component | Formula | Notes |
|---|---|---|
| `R_billing` | `+Σ client_rate_weekly(c)` for all placed candidates `c` | Weekly billing — varies by each candidate's composite rating |
| `C_salary` | `−Σ salary_weekly(c)` for all hired candidates `c` | Weekly salary burn — varies by each candidate's rating |
| `C_onboarding` | `−$2,000` per `hire_candidate` event | One-time, rating-independent |
| `C_severance` | `−2 × salary_weekly(c)` per `let_go_candidate` event | Rating-dependent — higher rated candidates cost more to sever |
| `C_expired` | `−Σ client_rate_weekly(roles)` × `weeks_remaining` per expired project | Opportunity cost at the rating-specific rate of the best-fit candidate |
| `C_churn` | `−estimated_LTV` per churned client | LTV computed from historical `client_rate_weekly` billing |
| `B_speed` | `+10% of weekly margin` if project sealed within 2 weeks of confirmation | Shaping bonus scales with candidate quality |

**Key identities (rating-dependent):**
- Margin per placed candidate: `salary_weekly × 0.25` (always 25%)
- Break-even weeks for a new hire: `$2,000 / margin_weekly`
  - Rating 5 candidate: `$2,000 / $625 ≈ 3.2 weeks`
  - Rating 3 candidate: `$2,000 / $408 ≈ 4.9 weeks`
  - Rating 1 candidate: `$2,000 / $265 ≈ 7.5 weeks`

---

## 9. Episode Structure

| Property | Value |
|---|---|
| Time step | 1 business week |
| Episode horizon | 52 steps (1 simulated business year) |
| Seed capital | Configurable (e.g. $50,000 default) |
| Reset | New random client pool, candidate market, zero placements |
| Terminal: bankruptcy | `cash_balance < 0` → episode ends immediately |
| Terminal: natural end | All projects filled, no new arrivals, step count at `T_max` |

**Curriculum suggestion (training stability):**
- Stage 1: 1 client, 1 developer_type, 1-role projects
- Stage 2: 3 clients, 3 developer_types, 2-role projects
- Stage 3: Full env — N clients, 5 developer_types, multi-role projects, stochastic demand

---

## 10. Observation Vector

Flattened observation fed to the policy network each step:

```python
obs = [
  # Agency financials
  cash_balance,
  current_profit,
  burn_rate,
  cash_runway_weeks,

  # Staffing metrics
  num_candidates_hired,
  num_candidates_placed,
  num_candidates_benched,
  num_candidates_in_interview,
  placement_rate,
  avg_time_to_place,

  # Demand signals
  *market_demand_vector,          # float per developer_type (5-dim)
  num_projects_pending,           # across all clients
  total_open_role_slots,          # sum of unfilled headcount across all open projects
  *project_deadline_histogram,    # bucketed urgency: [0-2wk, 3-5wk, 6+wk] counts

  # Client health
  num_active_clients,
  avg_client_satisfaction,
  num_clients_at_churn_risk,

  # Candidate urgency
  num_churn_risk_candidates,      # patience_remaining ≤ 2
  avg_weeks_on_bench,
]
```

**Variable-length sets** (per-project roles, per-candidate profiles) require one of:
- Fixed-size padded tensors with a mask
- Attention / set encoder (recommended for scalability)
- Graph Neural Network (projects as nodes, candidates as nodes, edges = valid matches)

---

## 11. Open Technical & Research Questions

### Environment Design

| ID | Question | Why It Matters |
|---|---|---|
| E1 | What is the contract duration per placement — fixed (26/52 wks) or client-terminated? | Determines when a placed candidate returns to bench and revenue ends |
| E2 | Is there a cap on simultaneous open projects per client? | Bounds client state space size |
| E3 | How many developer types and what is the adjacency/compatibility matrix? | Defines supply/demand mismatch dimensionality and matching action legality |
| E4 | How is client project arrival modelled — independent Poisson per client, or correlated (market cycles)? | Stochastic demand model is the primary source of exploration difficulty |
| E5 | Is interview outcome deterministic or probabilistic (pass/fail with noise)? | Adds pipeline uncertainty; agent must decide whether to re-interview |
| E6 | What happens if a placed candidate is let go mid-contract — client penalty, breach clause? | Defines full consequences of `let_go_candidate` on active placements |
| E7 | Is partial observability in scope (POMDP)? Does the agent see all client queues or only queried ones? | Determines MDP vs POMDP — large architecture and algorithm impact |
| E8 | Can the same candidate be re-placed after a contract ends, or do they re-enter the market? | Affects candidate lifecycle modelling and long-term bench strategy |

### Reward & Training

| ID | Question | Why It Matters |
|---|---|---|
| R1 | How do we handle sparse rewards from multi-role project sealing? Projects with 4+ roles may take many steps to fill before any revenue triggers | Need reward shaping, intrinsic curiosity, or hierarchical RL to avoid credit assignment failure |
| R2 | Should client LTV be estimated and included in the churn penalty, or left to the value function? | LTV in reward requires a model; leaving it to V(s) requires longer horizon training |
| R3 | What discount factor γ? Low γ ignores client satisfaction decay; high γ makes training unstable | Directly encodes how much the agent values long-term relationships |
| R4 | How do we prevent reward hacking — e.g. agent confirms projects it cannot fill, or repeatedly hires/fires to game cost accounting? | Requires action masking + constraint penalties + audit of degenerate policies |
| R5 | Single scalarised reward or multi-objective (profit vs satisfaction vs candidate welfare)? | Multi-objective RL opens a Pareto-front research question |

### Agent Architecture

| ID | Question | Why It Matters |
|---|---|---|
| A1 | How is the variable-length action space handled — candidates and projects change count each step? | Pointer networks, action masking over sets, or bipartite matching head needed |
| A2 | How many tool calls per step — unlimited, fixed budget, or hierarchical (macro + micro actions)? | Defines step semantics and computational cost per episode |
| A3 | Is this a single-step decision (classic RL) or a multi-step tool-use chain per episode step (LLM agent loop)? | If agent chains `find → interview → hire` within one step, architecture is closer to ReAct/LLM-agent than PPO |
| A4 | What is the encoding for multi-role projects — fixed-size padding, attention over role sets, or GNN? | Directly impacts policy expressiveness and sample efficiency |
| A5 | Which RL algorithm is the baseline — PPO (discrete), SAC (continuous), Decision Transformer, or LLM-as-agent? | Architectural fork; agentic LLM-based RL needs a different training loop than classic DNN-RL |
| A6 | How is the bipartite matching between candidates and project roles handled inside the policy — end-to-end learned or solved as a constrained optimisation sub-problem? | Core research question given multi-role, multi-candidate assignment combinatorics |

### Simulation & Evaluation

| ID | Question | Why It Matters |
|---|---|---|
| S1 | What are the baseline policies? Suggestions: random, greedy (hire whenever demand > bench), rule-based (hire only if open roles > bench slots) | Without baselines it is impossible to assess whether RL is adding value |
| S2 | How is client/candidate behaviour calibrated — real salary distributions, project arrival rates, satisfaction decay functions? | Sim-to-real gap is the main risk for eventual deployment utility |
| S3 | Is curriculum learning required for stable cold-start training given sparse multi-role rewards? | Large action spaces + sparse rewards often make naive PPO fail without staged difficulty |
| S4 | What evaluation metrics beyond profit? Suggested: time-to-fill, bench rate, client retention rate, candidate churn rate, match quality distribution | Profit alone may mask pathological or unethical policies |
| S5 | Is there a human-in-the-loop mode where a human approves certain actions (e.g. letting go a candidate)? | Required if this is a decision-support tool rather than a fully autonomous system |
| S6 | How do we detect and penalise degenerate policies during training (e.g. always pass on projects, never hire)? | Policy collapse is a known failure mode in sparse reward business envs |

---

## 12. LLM Calls Within the Environment

Certain environment transitions are too semantically rich to model with a probability
distribution — the outcome depends on *context*, *nuance*, and *reasoning* that a fixed
`p(outcome)` cannot capture faithfully. In these cases, the environment delegates to an
**LLM judge** that returns a structured verdict.

All LLM calls follow the same stub interface:

```python
def llm_call(prompt: str, schema: dict) -> dict:
    """
    Stub: replace with real Anthropic API call.
    Returns a structured JSON response conforming to `schema`.
    In stub mode, returns schema defaults or random-seeded plausible values.
    """
    # TODO: wire to anthropic.messages.create(model="claude-sonnet-4-20250514", ...)
    raise NotImplementedError("LLM stub — not yet wired")
```

---

### 12.1 `llm_interview(candidate, job_description) → InterviewResult`

**Triggered by:** `interview_candidate(candidate_id)`

**Why not a distribution:** A candidate's interview performance depends on their
background, the role's requirements, and conversational dynamics. A flat
`Beta(α, β)` over skill_score loses the per-role context entirely.

```python
@dataclass
class InterviewResult:
    base_rating:      int    # 1–5
    technical_score:  float  # 0.0–1.0  domain knowledge
    communication:    float  # 0.0–1.0  clarity, professionalism
    culture_fit:      float  # 0.0–1.0  values alignment
    red_flags:        list[str]  # e.g. ["gap in employment", "vague on system design"]
    summary:          str    # 2–3 sentence interviewer note
    proceed:          bool   # recommend advancing to hire decision

def llm_interview(candidate: Candidate, job_description: str) -> InterviewResult:
    prompt = f"""
    You are a senior technical interviewer at a staffing agency.
    Candidate profile: {candidate}
    Role being considered: {job_description}

    Conduct a simulated interview and return a structured assessment.
    Be realistic — not all candidates perform well.
    """
    # STUB
    return InterviewResult(
        base_rating=3,
        technical_score=0.65,
        communication=0.70,
        culture_fit=0.60,
        red_flags=[],
        summary="Solid mid-level candidate. Adequate for standard backend roles.",
        proceed=True,
    )
```

---

### 12.2 `llm_project_fit(candidate, role, project_context) → FitResult`

**Triggered by:** `match_candidate_to_project(candidate_id, project_id, role_id)`

**Why not a distribution:** Project fit depends on the *specific* role requirements,
team context, and client industry — a generic skill_score match cannot capture whether
a backend engineer with fintech experience is right for a healthtech ML pipeline role.

```python
@dataclass
class FitResult:
    project_fit_rating: int    # 1–5
    composite_rating:   float  # 0.4×base_rating + 0.6×project_fit_rating
    fit_rationale:      str    # 2–3 sentence reasoning
    risk_flags:         list[str]  # e.g. ["no prior healthtech exposure"]
    client_satisfaction_delta: float  # expected satisfaction impact: −1.0 to +1.0

def llm_project_fit(candidate: Candidate, role: Role,
                    project_context: dict) -> FitResult:
    prompt = f"""
    You are evaluating whether a candidate is a good fit for a specific project role.
    Candidate: {candidate}
    Role: {role}
    Project context (client industry, team, tech stack): {project_context}

    Rate the fit on a 1–5 scale and explain your reasoning.
    Flag any risks the agency should be aware of.
    """
    # STUB
    composite = round(0.4 * candidate.base_rating + 0.6 * 3, 2)
    return FitResult(
        project_fit_rating=3,
        composite_rating=composite,
        fit_rationale="Acceptable match. Candidate meets minimum requirements.",
        risk_flags=[],
        client_satisfaction_delta=0.0,
    )
```

---

### 12.3 `llm_salary_negotiation(candidate, offer, market_context) → NegotiationResult`

**Triggered by:** `negotiate_salary(candidate_id, offer)`

**Why not a distribution:** Acceptance probability is not just a function of offer vs.
expectation. It depends on the candidate's competing offers, urgency, how the offer was
framed, and perceived career growth — none of which a sigmoid over salary delta captures.

```python
@dataclass
class NegotiationResult:
    accepted:           bool
    counter_offer:      float | None  # if not accepted, candidate's counter
    acceptance_reason:  str
    patience_impact:    int   # Δ patience_remaining (negative = more urgent to decide)

def llm_salary_negotiation(candidate: Candidate, offer: float,
                            market_context: dict) -> NegotiationResult:
    prompt = f"""
    You are simulating a candidate's response to a salary offer from a staffing agency.
    Candidate: {candidate}
    Offer (weekly): ${offer}
    Candidate's minimum expectation (weekly): ${candidate.salary_expectation}
    Market context: {market_context}

    Would the candidate accept, reject, or counter? Reason from their perspective.
    """
    # STUB
    accepted = offer >= candidate.salary_expectation
    return NegotiationResult(
        accepted=accepted,
        counter_offer=None if accepted else candidate.salary_expectation * 1.05,
        acceptance_reason="Offer meets expectation." if accepted else "Below market rate.",
        patience_impact=-1 if not accepted else 0,
    )
```

---

### 12.4 `llm_client_satisfaction(client, event, history) → SatisfactionUpdate`

**Triggered by:** After every material event — project sealed, project expired,
partial fill update, adjacent match used, extension requested.

**Why not a distribution:** Client satisfaction is a *relationship* — it has memory,
context, and industry norms. A fintech client reacts differently to a delayed ML hire
than a startup client does. A single scalar delta function misses this entirely.

```python
@dataclass
class SatisfactionUpdate:
    new_score:      float        # 0.0–1.0
    delta:          float        # change from prior score
    churn_risk:     bool
    client_message: str          # simulated client feedback (e.g. "We need the role filled ASAP")
    lTV_impact:     float        # estimated change in expected future revenue

def llm_client_satisfaction(client: Client, event: dict,
                             history: list[dict]) -> SatisfactionUpdate:
    prompt = f"""
    You are simulating a client's satisfaction with a staffing agency.
    Client profile: {client}
    Recent event: {event}
    Relationship history (last 5 events): {history[-5:]}

    Update the satisfaction score (0–1) and generate realistic client feedback.
    Consider industry norms and relationship history.
    """
    # STUB
    delta = 0.05 if event.get("type") == "project_sealed" else -0.1
    new_score = max(0.0, min(1.0, client.satisfaction_score + delta))
    return SatisfactionUpdate(
        new_score=new_score,
        delta=delta,
        churn_risk=new_score < 0.3,
        client_message="Stub: client feedback not yet wired.",
        lTV_impact=delta * 10000,
    )
```

---

### 12.5 `llm_candidate_leave(candidate, agency_context) → LeaveDecision`

**Triggered by:** Each step, for candidates with `patience_remaining ≤ 2`.

**Why not a distribution:** Whether a benched candidate leaves depends on their
financial situation, competing offers in the market, and how the agency has treated
them — not just a countdown timer.

```python
@dataclass
class LeaveDecision:
    leaves:   bool
    reason:   str    # e.g. "Accepted competing offer at higher rate"
    patience_remaining: int  # updated value

def llm_candidate_leave(candidate: Candidate,
                         agency_context: dict) -> LeaveDecision:
    prompt = f"""
    You are simulating whether a candidate decides to leave a staffing agency's bench.
    Candidate: {candidate}
    Weeks on bench: {candidate.weeks_on_bench}
    Agency context (market demand, competing offers, agency's recent behaviour): {agency_context}

    Does the candidate leave? Give a realistic reason.
    """
    # STUB
    leaves = candidate.patience_remaining <= 0
    return LeaveDecision(
        leaves=leaves,
        reason="Patience exhausted — accepted competing offer." if leaves else "Still waiting.",
        patience_remaining=max(0, candidate.patience_remaining - 1),
    )
```

---

### 12.6 LLM Call Summary

| Call | Trigger | Replaces | Output |
|---|---|---|---|
| `llm_interview` | `interview_candidate()` | `Beta` distribution over skill_score | `base_rating`, red flags, proceed |
| `llm_project_fit` | `match_candidate_to_project()` | Fixed match_score formula | `project_fit_rating`, satisfaction delta |
| `llm_salary_negotiation` | `negotiate_salary()` | Sigmoid over salary delta | accept/counter, patience impact |
| `llm_client_satisfaction` | Every material event | Scalar delta function | new score, churn risk, client message |
| `llm_candidate_leave` | Each step, low-patience candidates | Countdown timer | leave decision, reason |

### 12.7 Stub vs. Live Mode

```python
LLM_MODE = "stub"  # "stub" | "live"

def llm_call_router(fn_name: str, **kwargs):
    if LLM_MODE == "stub":
        return STUBS[fn_name](**kwargs)
    elif LLM_MODE == "live":
        return LIVE[fn_name](**kwargs)   # wired to Anthropic API
```

> When switching from stub → live, **all LLM outputs must be logged and cached**
> to ensure episode reproducibility for RL training. Stochastic LLM responses
> will break trajectory replay unless seeded or cached per episode.

---

## Appendix: Developer Type Reference

| Type | Adjacency (partial match allowed) |
|---|---|
| `backend` | `fullstack`, `ml_engineer` |
| `frontend` | `fullstack` |
| `fullstack` | `backend`, `frontend` |
| `ml_engineer` | `backend` |
| `devops` | *(none — specialist, no adjacency)* |

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| **Sealed** | A project where all roles are 100% filled; revenue begins |
| **Benched** | A hired candidate not currently assigned to any project |
| **Placement rate** | `num_placed / num_hired` — primary efficiency KPI |
| **Churn** | A client leaving due to low satisfaction; stops submitting projects |
| **match_score** | 0–1 quality of a candidate-role assignment; affects satisfaction |
| **Patience window** | Max steps a candidate waits before leaving the agency |
| **Deadline window** | Max steps before an unfilled project expires |
| **LTV** | Lifetime Value — estimated future revenue from a client |
| **Burn rate** | Weekly salary outflow across all hired candidates |
| **Cash runway** | `cash_balance / burn_rate` — weeks until bankruptcy if no revenue |