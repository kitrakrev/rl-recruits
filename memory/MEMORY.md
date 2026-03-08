# Project Memory — rl_recruiter (Staffing Agency OpenEnv)

## Project Type
OpenEnv hackathon RL environment. Staffing agency agent maximises profit over 52-step episode.

## Key Files
- `server/staffing_environment.py` — `StaffingAgencyEnvironment(MCPEnvironment)` — main env
- `server/app.py` — `create_app()` factory → FastAPI OpenEnv server
- `client.py` — `StaffingAgencyEnv(MCPToolClient)` — client wrapper
- `env/config.py` — all tunable params (curriculum_stage, llm_mode, economics)
- `env/models.py` — Candidate, Role, Project, Client dataclasses
- `env/llm.py` — LLMRouter (stub + live Anthropic API), 5 LLM calls
- `env/simulation.py` — world dynamics: Poisson arrivals, patience, deadlines, churn
- `training/train_grpo.py` — GRPO training + dry-run reward curves
- `tests/test_env.py` — 25 tests for all 17 tools
- `plan.md` — implementation checklist
- `guess.md` — documented assumptions

## Rules (from `rules` file)
- Python3 + uv venv
- LLM stub if no key (static/probabilistic fallback)
- Assume missing details → document in `guess.md`
- Maintain `plan.md` with checkbox progress

## Setup
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest tests/ -v
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
# Dry-run training (no GPU):
uv run python training/train_grpo.py --dry_run --num_episodes 90
```

## OpenEnv Architecture
- Base: `MCPEnvironment` from `openenv.core.env_server.mcp_environment`
- Tools: registered via `@mcp.tool` on `FastMCP` instance
- Server: `create_app(StaffingAgencyEnvironment, CallToolAction, CallToolObservation)`
- Client: `StaffingAgencyEnv(MCPToolClient)` from `openenv.core.mcp_client`
- State: `StaffingState(State)` — Pydantic model
- Action: `CallToolAction(tool_name, arguments)` from `openenv.core.env_server.mcp_types`

## LLM Mode
- Default: `stub` (no API key needed)
- Live: set `LLM_MODE=live` + `ANTHROPIC_API_KEY=sk-...`
- Model: `claude-sonnet-4-6`

## Observation Space
25-dim float32: [cash, profit, burn, runway | hired, placed, benched, pipeline, placement_rate, avg_ttp | 5×demand, n_projects, open_slots, 3×deadline_hist | n_active_clients, avg_sat, n_churn_risk | n_churn_candidates, avg_bench_weeks]

## Economics (key numbers — updated)
- Interview cost: −$500 per candidate screened (returned in reward field)
- Salary is DYNAMIC: salary_expectation = base × role_multiplier × skill_modifier × ±20% RNG
  → base: Junior $75k, Mid $110k, Senior $150k; multipliers: frontend 1.0, backend 1.05, devops 1.15, ML 1.3
- Bill rate is VARIABLE per role: client pays $130k–$300k+/yr (1.3×–2.0× base × scarcity)
- TRUE MARGIN = role.bill_rate_weekly − candidate.salary_weekly (not a fixed 25%)
  → A losing placement is possible if salary > bill_rate
- Onboarding: −$2,000 one-time; Severance: 2× weekly salary
- Bench burn: candidate.salary_weekly per week (dynamic)

## train_grpo.py Iterative Loop (implemented)
- PHASE 1 (Rollout): play N full 52-step episodes with current model → record prompts/completions/final_profit
- PHASE 2 (Dataset): assign final_profit as Monte Carlo return to EVERY step in trajectory
- PHASE 3 (Train): run GRPOTrainer for 1 epoch on this dataset
- PHASE 4 (Repeat): loop back. Graphs plot true episode profit over training iterations.
- reward_fn_mc_profit receives mc_return column from dataset (final_profit / 1000 for stability)
