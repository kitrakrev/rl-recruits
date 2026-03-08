# Staffing Agency — OpenEnv RL Environment

> **Hackathon Theme:** Multi-Agent Interactions + Long-Horizon Planning (Scale AI / Mercor sub-theme)
>
> An LLM agent acts as a **Staffing Agency CEO** managing multiple clients and candidates
> over a 52-week simulated business year. The agent must balance bench costs, candidate
> patience, project deadlines, and client satisfaction to maximise profit.

---

## Why This Environment Is Interesting

| Property | Description |
|---|---|
| **Multi-Actor** | Agent manages N clients (demand) + M candidates (supply) simultaneously |
| **Long-Horizon** | 52 steps; projects with 3+ roles take many steps to seal → sparse reward |
| **Emergent Strategy** | Over-hiring bleeds cash; under-hiring loses clients → narrow optimal band |
| **LLM-Graded Transitions** | Interview, fit assessment, salary negotiation, client satisfaction via LLM |
| **OpenEnv Native** | MCPEnvironment + FastMCP tools + create_app — deployable to HF Spaces |

---

## Quick Start

```bash
# 1. Create venv with uv
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install
uv pip install -e ".[dev]"

# 3. Run tests (no server needed)
uv run pytest tests/ -v

# 4. Start OpenEnv server
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# 5. Test the server
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"seed": 42}'
```

---

## Training (Dry Run — no GPU needed)

```bash
# Simulate reward curves comparing random vs greedy vs optimal policy
uv run python training/train_grpo.py --dry_run --num_episodes 90

# Outputs:
#   training/reward_curves.png      ← before/after reward comparison plot
#   training/metrics_summary.json   ← mean profit, positive rate per policy
```

## Training (Full GRPO — requires GPU)

```bash
# Install training dependencies
uv pip install -e ".[train]"

# Start the environment server in one terminal
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Train in another terminal
uv run python training/train_grpo.py \
    --env_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --num_episodes 200 \
    --output_dir training/checkpoints
```

---

## OpenEnv Architecture

```
StaffingAgencyEnvironment
  └── MCPEnvironment (openenv-core)
        └── FastMCP tools (17 tools)
              ├── GET tools (7):  get_agency_state, get_client_state, ...
              └── EXECUTE tools (10): find_candidate, interview_candidate, hire_candidate, ...

server/app.py
  └── create_app(StaffingAgencyEnvironment, CallToolAction, CallToolObservation)
        └── FastAPI with /reset, /step, /state, /health, /schema, /ws (WebSocket)

client.py
  └── StaffingAgencyEnv(MCPToolClient)
        └── reset(), list_tools(), call_tool(), step(), sync()
```

---

## Key Economics (what the agent must learn)

| Scenario | Weekly P&L |
|---|---|
| Rating-5 candidate placed | +$625 margin |
| Rating-3 candidate placed | +$408 margin |
| Rating-5 candidate benched | −$2,500 burn |
| Rating-1 candidate benched | −$1,058 burn |
| New hire onboarding | −$2,000 (one-time) |
| Severance | −2× weekly salary |
| Project expiry | −opportunity cost |
| Client churn | −$50,000 LTV |

**Break-even:** A Rating-5 hire pays back onboarding in ~3.2 weeks if placed.

---

## Available Tools (17 total)

**GET (observation):**
| Tool | Description |
|---|---|
| `get_agency_state` | Cash, revenue, costs, profit, burn, runway |
| `get_client_state` | Per-client or all-client satisfaction, projects |
| `get_candidate_state` | Pipeline, bench, churn risk candidates |
| `get_project_details` | Roles, deadline, fill status for one project |
| `get_candidate_profile` | Full profile of one candidate |
| `get_market_demand` | Open role slots by developer type |
| `get_financial_summary` | P&L snapshot |

**EXECUTE (actions):**
| Tool | Description |
|---|---|
| `find_available_projects` | Discover all open projects |
| `confirm_project` | Commit to a project (satisfaction boost) |
| `find_candidate` | Search market, optionally by type |
| `interview_candidate` | Set base_rating, move to pipeline |
| `hire_candidate` | Put on payroll (−$2,000 onboarding) |
| `negotiate_salary` | Adjust offer; affects acceptance |
| `match_candidate_to_project` | Place candidate; seals project when all roles filled |
| `let_go_candidate` | Remove from payroll (−2× salary severance) |
| `request_project_extension` | Buy time (satisfaction cost) |
| `pass_on_project` | Decline a project to avoid expiry penalty |

---

## LLM Mode

```bash
# Stub (default — no API key, uses probabilistic responses)
LLM_MODE=stub uv run python -m uvicorn server.app:app --port 8000

# Live (uses claude-sonnet-4-6 for rich semantic evaluations)
LLM_MODE=live ANTHROPIC_API_KEY=sk-... uv run python -m uvicorn server.app:app --port 8000
```

---

## Curriculum Stages

| Stage | Clients | Dev Types | Max Roles/Project |
|---|---|---|---|
| 1 (easy) | 1 | 1 (backend) | 1 |
| 2 (medium) | 3 | 3 | 2 |
| 3 (full) | 3+ | 5 | 3 |

Set in `Config(curriculum_stage=1)`.

---

## File Structure

```
rl_recruiter/
├── env/
│   ├── config.py            ← All parameters (economics, curriculum, LLM)
│   ├── models.py            ← Candidate, Role, Project, Client dataclasses
│   ├── llm.py               ← LLMRouter: 5 LLM calls, stub + live
│   └── simulation.py        ← World dynamics: arrivals, deadlines, churn
├── server/
│   ├── staffing_environment.py  ← StaffingAgencyEnvironment(MCPEnvironment)
│   └── app.py               ← create_app() → FastAPI OpenEnv server
├── training/
│   └── train_grpo.py        ← GRPO training + dry-run reward curves
├── tests/
│   └── test_env.py          ← 25 tests for all tools
├── client.py                ← StaffingAgencyEnv(MCPToolClient)
├── pyproject.toml
├── plan.md                  ← Implementation checklist
└── guess.md                 ← Documented assumptions
```

---

## Judging Criteria Alignment

| Criterion | How We Address It |
|---|---|
| **Environment Innovation (40%)** | Multi-actor (clients + candidates), sparse multi-role sealing, LLM-graded transitions, 52-step horizon — novel HR/business domain |
| **Storytelling (30%)** | Clear CEO framing, economic tensions explained, rich tool names with docstrings |
| **Reward Improvement (20%)** | `train_grpo.py --dry_run` shows random→greedy→optimal reward curves |
| **Training Pipeline (10%)** | Full TRL GRPO pipeline with 3 reward functions (profit, format, placement) |
