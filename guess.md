# Assumptions & Guesses

All items here are assumptions made due to missing or ambiguous spec details.

## Episode & Environment
- `T_patience` (candidate patience window): **8 steps** (8 weeks). Reasonable for a job seeker.
- `T_deadline` (project deadline window): **6 steps** (6 weeks) default per project.
- Number of clients: **3** in full env, **1** in Stage 1 curriculum.
- Number of simultaneous open projects per client: **capped at 3**.
- Seed capital: **$50,000** (stated in spec as default).
- Max candidates in market at any time: **20** across all developer types.
- Poisson λ for project arrivals per client per step: **0.5** (roughly 1 project every 2 weeks).
- Contract duration per placement: **26 weeks** (6 months, stated as default).

## Candidate Generation
- `salary_expectation` is sampled as `salary_weekly × U(0.85, 1.05)` — candidate wants roughly market rate ±15%.
- `skill_score` is sampled from `Beta(2, 2)` shifted to [0.3, 1.0] — most candidates are mid-range.
- Developer type distribution: uniform across 5 types for simplicity.
- Seniority distribution: 40% junior, 40% mid, 20% senior.

## Client Generation
- Industry assigned randomly from: `["fintech", "healthtech", "ecommerce", "saas", "logistics"]`.
- Initial `satisfaction_score`: **0.75** (starts reasonably satisfied).
- `churn_threshold`: **0.3** — client churns if satisfaction falls below this.
- LTV estimate for churn penalty: **$50,000** per churned client (rough proxy for 26 wks × avg billing).

## Project Generation
- Role `min_skill_score` sampled from `U(0.3, 0.8)` — variety from easy to hard roles.
- `headcount` per role: **1–2** (sampled), with at most **3 roles** per project.
- Project deadline: `U(4, 10)` steps at creation time.

## Matching
- Adjacency matrix as defined in spec Appendix.
- Seniority compatibility: junior fills junior only; mid fills junior or mid; senior fills any.

## LLM Stubs
- Stub mode returns static plausible values with small Gaussian noise.
- `LLM_MODE = "stub"` by default — no API key required to run.
- Live mode wired to `claude-sonnet-4-6` model when `ANTHROPIC_API_KEY` is set.

## OpenEnv Interface
- Using `openenv` package conventions: `reset()`, `step(action)`, `render()`, `action_space`, `observation_space`.
- Action is a dict (tool call) with `tool_name` and `params`.
- Observation is a flat float32 numpy array (25-dim as per Section 10).
- `info` dict returned by `step()` contains full structured state for debugging.

## Reward Scaling
- Raw dollar rewards are **divided by 1000** for numerical stability in neural net training.

## Not Implemented (out of scope for v1)
- POMDP partial observability mode (E7).
- Human-in-the-loop mode (S5).
- Multi-objective reward (R5).
- GNN policy architecture (A4/A6) — left to researcher.
