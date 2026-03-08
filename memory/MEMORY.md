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

## Economics (key numbers)
- Rating-5 placed: +$625/wk margin
- Rating-5 benched: −$2,500/wk
- Onboarding: −$2,000 one-time
- Severance: 2× weekly salary
- Margin always 25% of client rate
