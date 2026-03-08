# Implementation Plan — Staffing Agency OpenEnv

## Status Legend
- [ ] Pending
- [x] Done
- [-] In Progress

---

## Phase 1 — Project Scaffold
- [x] `pyproject.toml` with uv/pip metadata + openenv-core dependency
- [x] uv venv setup instructions in README
- [x] Directory structure created

## Phase 2 — Core Data Models (`env/models.py`)
- [x] `Candidate` dataclass
- [x] `Role` dataclass
- [x] `Project` dataclass
- [x] `Client` dataclass
- [x] `AgencyState` dataclass

## Phase 3 — LLM Layer (`env/llm.py`)
- [x] `LLMRouter` with stub/live toggle (via Config.llm_mode)
- [x] `llm_interview` stub + live
- [x] `llm_project_fit` stub + live
- [x] `llm_salary_negotiation` stub + live
- [x] `llm_client_satisfaction` stub + live
- [x] `llm_candidate_leave` stub + live

## Phase 4 — OpenEnv MCPEnvironment (`server/staffing_environment.py`)
- [x] `StaffingAgencyEnvironment(MCPEnvironment)` — proper OpenEnv base class
- [x] `StaffingState(State)` — Pydantic state model
- [x] `reset()` — generate clients, candidates, initial observation
- [x] `step()` — routes MCP tool calls + world tick + reward
- [x] `state` property
- [x] All 17 tools registered via `@mcp.tool` (FastMCP)
- [x] Reward function (Section 8 of spec)
- [x] Episode termination logic (bankruptcy + 52 steps)

## Phase 5 — Simulation Logic (`env/simulation.py`)
- [x] Stochastic project arrival (Poisson)
- [x] Candidate patience decay + LLM leave decisions
- [x] Project deadline countdown + expiry penalty
- [x] Client satisfaction updates (LLM-driven)
- [x] Contract completion / bench return
- [x] Client churn logic + LTV penalty

## Phase 6 — OpenEnv Server (`server/app.py`)
- [x] `create_app()` using openenv-core factory
- [x] `CallToolAction` / `CallToolObservation` as action/observation types
- [x] All OpenEnv endpoints: /reset, /step, /state, /health, /schema, /ws

## Phase 7 — Client (`client.py`)
- [x] `StaffingAgencyEnv(MCPToolClient)` — client wrapper
- [x] Sync + async usage

## Phase 8 — Training Script (`training/train_grpo.py`)
- [x] System prompt for staffing CEO agent
- [x] TOOL: / PARAMS: format parsing
- [x] Three reward functions: profit, format, placement
- [x] Dry-run mode: random vs greedy vs optimal_heuristic policy comparison
- [x] Reward curve plotting (training/reward_curves.png)
- [x] Metrics JSON (training/metrics_summary.json)
- [x] Full TRL GRPO training loop (GPU path)

## Phase 9 — Tests (`tests/test_env.py`)
- [x] 32 tests covering all 17 tools
- [x] Full hire-place pipeline test
- [x] Illegal match test
- [x] Bankruptcy termination test
- [x] 52-step rollout test
- [x] **32/32 tests PASSING** ✓

## Phase 10 — Full OpenEnv Compliance (per openenv.md)
- [x] `openenv-core` installed and confirmed working
- [x] `MCPEnvironment` base class used correctly
- [x] `@mcp.tool()` decorator syntax correct (FastMCP 3.x)
- [x] `ListToolsAction` returns proper `ListToolsObservation` with `.tools`
- [x] `CallToolAction` routes to correct FastMCP tool, returns `.data` dict
- [x] GET tools (read-only, no tick) vs EXECUTE tools (tick world) separation
- [x] `create_app(StaffingAgencyEnvironment, CallToolAction, CallToolObservation)` in server/app.py
- [x] `models.py` — typed `StaffingAction`, `StaffingObservation`, `StaffingState`
- [x] `client.py` — `StaffingAgencyEnv(EnvClient[...])` with `_step_payload`, `_parse_result`, `_parse_state`
- [x] `openenv.yaml` — manifest with name, entry_point, tools list, curriculum config
- [x] `server/Dockerfile` — standard OpenEnv Dockerfile pattern
- [x] **32/32 tests PASSING** ✓

## Phase 11 — Live Training Dashboard (`ui/dashboard.py`)
- [x] Gradio dark-theme dashboard with 6 tabs
- [x] KPI strip: step, cash, profit, cumulative reward, placed, benched, placement%, runway
- [x] Tab 1 — Live Charts: financials, satisfaction, candidate pipeline (stacked), burn/runway, market demand, placement rate
- [x] Tab 2 — Agent Analytics: reward history, tool-call frequency, avg reward per tool, agent action history table, episode history table, candidate lifecycle events
- [x] Tab 3 — Candidates: full table with status, skill, rating, margin, patience, project, contract weeks
- [x] Tab 4 — Clients & Projects: client satisfaction bars, churn risk, project fill cards
- [x] Tab 5 — Manual Override: tool dropdown with param hints, JSON params, state override (cash, satisfaction)
- [x] Tab 6 — Episode Control: reset with seed, quick-action buttons, event log
- [x] `_override_cash` + `_override_satisfaction` admin tools added to `StaffingAgencyEnvironment`
- [x] Override tools added to `_GET_TOOLS` (no episode step cost)
- [x] `gradio` + `plotly` + `requests` added to `pyproject.toml` [ui] extras
- [x] `--start_server` flag to auto-launch env server before UI
- [x] Auto-refresh via `gr.Timer`, configurable `--poll` interval

---

_Last updated: 2026-03-07_
