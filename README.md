# Staffing Agency — OpenEnv RL Environment

> **Hackathon Theme:** Multi-Agent Interactions + Long-Horizon Planning (Scale AI / Mercor sub-theme)
>
> An LLM agent acts as a **Staffing Agency CEO** managing multiple clients and candidates
> over a 52-week simulated business year. The agent must balance bench costs, candidate
> patience, project deadlines, and client satisfaction to maximise profit using real
> tool calls against a live environment server — no fabricated rewards.

---

## Why This Environment Is Interesting

| Property | Description |
|---|---|
| **Multi-Actor** | Agent manages N clients (demand) + M candidates (supply) simultaneously |
| **Long-Horizon** | 52 steps; multi-role projects take many weeks to seal → sparse reward |
| **Emergent Strategy** | Over-hiring bleeds cash; under-hiring loses clients → narrow optimal band |
| **Real Rewards** | Every reward comes from env.step() — interview costs, bench burn, billing margins, expiry penalties |
| **LLM-Graded Transitions** | Interview, fit, salary negotiation, client satisfaction via LLM (stub or live) |
| **OpenEnv Native** | MCPEnvironment + FastMCP + create_app — deployable to HF Spaces |
| **Live Config API** | PATCH /config/env hot-patches environment params without restarting the server |

---

## Quick Start

```bash
# 1. Create venv
uv venv .venv && source .venv/bin/activate

# 2. Install
uv pip install -e ".[dev]"

# 3. Run tests (no server needed)
uv run pytest tests/ -v

# 4. Start the environment server
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# 5. Check health + config
curl http://localhost:8000/health
curl http://localhost:8000/config
```

---

## Training

### Dry Run (no GPU — validates the full HTTP stack)

```bash
uv run python training/train_grpo.py --dry_run --num_episodes 90

# Outputs:
#   training/reward_curves.png      ← random vs greedy vs optimal reward curves
#   training/metrics_summary.json   ← mean profit, positive rate per policy
```

### Full Training (GPU required)

```bash
# Install training dependencies
uv pip install -e ".[train]"

# Terminal 1 — environment server
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — training
uv run python training/train_grpo.py \
    --env_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --num_episodes 200 \
    --output_dir training/checkpoints \
    --wandb_api_key YOUR_KEY
```

### Training from a YAML config file

```bash
# training/config.yaml
# model_name: Qwen/Qwen2.5-1.5B-Instruct
# learning_rate: 3e-6
# gamma: 0.99
# kl_coeff: 0.05
# num_episodes: 500
# wandb_project: myorg/staffing-agent

uv run python training/train_grpo.py --config training/config.yaml
# CLI flags override YAML values when both are provided
uv run python training/train_grpo.py --config training/config.yaml --num_episodes 100
```

---

## Config API (live hot-patch)

The environment server exposes three config endpoints that update parameters
**without restarting the server**. Changes take effect for the next episode reset.

```bash
# View full config (env + training defaults)
curl http://localhost:8000/config

# View only environment config
curl http://localhost:8000/config/env

# View training hyperparameter defaults
curl http://localhost:8000/config/training

# Switch to curriculum stage 2 (live)
curl -X PATCH http://localhost:8000/config/env \
     -H "Content-Type: application/json" \
     -d '{"curriculum_stage": 2, "num_clients": 3, "max_roles_per_project": 2}'

# Relax penalties during early training
curl -X PATCH http://localhost:8000/config/env \
     -H "Content-Type: application/json" \
     -d '{"passive_streak_threshold": 6, "repeat_call_penalty": -50.0}'
```

All configurable fields are documented in `env/config.py`.

---

## Reward Flow (how rewards reach the training loop)

```
env.core.tool_*(...)
  → returns {"reward": X, "success": True, ...}
        ↓
_register_tools wrapper
  → env._last_tool_reward = X   # store before stripping
  → pops "reward" key           # agent never sees it in conversation
        ↓
staffing_environment.step()
  → tool_reward = self._last_tool_reward
  → total_reward = tool_reward + passive_penalty + repeat_penalty
  → CallToolObservation(reward=total_reward)
        ↓
client._parse_result()
  → result.reward = total_reward
        ↓
rollout_full_episode()
  → step_rewards.append(result.reward)
        ↓
reinforce.train_grpo()
  → discounted returns → advantages → policy gradient update
```

**Key rewards per action:**

| Action | Reward |
|---|---|
| `interview_candidate` | −$500 interview cost |
| `hire_candidate` | −$2,000 onboarding cost |
| `let_go_candidate` | −2× weekly salary (severance) |
| `match_candidate_to_project` | small speed bonus if sealed within 2 weeks |
| `advance_week` | **main signal**: +margin per placed candidate, −bench burn, −expiry penalties, −client LTV if churn |
| Consecutive passive GET calls | −$50/turn after 3 free turns |
| Calling same tool twice in a row | −$100 |

---

## OpenEnv Architecture

```
StaffingAgencyEnvironment (server/staffing_environment.py)
  └── MCPEnvironment (openenv-core)
        └── FastMCP tools (19 tools registered)
              ├── GET tools (7):    get_agency_state, get_client_state, ...
              └── EXECUTE tools (12): find_candidate, interview_candidate,
                                      hire_candidate, advance_week, ...

server/app.py
  └── create_app(StaffingAgencyEnvironment, CallToolAction, CallToolObservation)
        └── FastAPI with /reset /step /state /health /config /config/env /config/training /ws

client.py (StaffingAgencyEnv)
  └── EnvClient[StaffingAction, StaffingObservation, StaffingState]
        └── reset(), step(), state()  — sync + async
```

---

## Training Architecture

```
training/
├── train_grpo.py    ← Entry point: parse_args() + dispatch to dry_run or reinforce
├── reinforce.py     ← REINFORCE-GRPO loop: rollout → returns → update
├── rollout.py       ← rollout_full_episode(): live 52-week env interaction
├── prompts.py       ← SYSTEM_PROMPT, TOOLS schema, parse_tool_call()
├── policies.py      ← Heuristic baselines: policy_random / greedy / optimal
├── dry_run.py       ← dry_run_simulate(): GPU-free validation via heuristic policies
└── metrics.py       ← plot_reward_curves(), save_metrics()
```

---

## Available Tools (19 total)

**GET (observation only — no reward, no state change):**

| Tool | Description |
|---|---|
| `get_agency_state` | Cash, revenue, costs, profit, burn, runway |
| `get_client_state` | Per-client or all-client satisfaction, projects |
| `get_candidate_state` | Pipeline, bench, churn risk candidates |
| `get_project_details` | Roles, deadline, fill status for one project |
| `get_candidate_profile` | Full profile of one candidate |
| `get_market_demand` | Open role slots by developer type |
| `get_financial_summary` | P&L snapshot |

**EXECUTE (carry reward, mutate state):**

| Tool | Reward Signal | Description |
|---|---|---|
| `find_available_projects` | 0 | Discover all open projects |
| `confirm_project` | 0 | Commit to a project (client satisfaction boost) |
| `find_candidate` | 0 | Search market by developer type |
| `interview_candidate` | **−$500** | Screen a candidate; reveals skills and salary |
| `hire_candidate` | **−$2,000** | Put on payroll (onboarding cost) |
| `negotiate_salary` | 0 | Adjust salary offer before hiring |
| `match_candidate_to_project` | **+speed bonus** | Place candidate; seals project when all roles filled |
| `let_go_candidate` | **−2× salary** | Remove from payroll (severance) |
| `request_project_extension` | 0 | Buy deadline time (satisfaction cost) |
| `pass_on_project` | 0 | Decline project (avoids expiry penalty) |
| `advance_week` | **main P&L** | Tick world: billing, bench burn, expiry, churn |

---

## Key Economics

| Scenario | Weekly Impact |
|---|---|
| Candidate placed (e.g., $3k bill − $1.4k salary) | +$1,600/wk margin |
| Candidate benched (salary still owed) | −salary/wk burn |
| New hire | −$2,000 one-time |
| Severance | −2× weekly salary |
| Project expiry (unfilled roles) | −large penalty per unfilled slot |
| Client churn (satisfaction < 0.3) | −$50,000 LTV |

**Break-even:** A $1,600/wk margin hire pays back $2,000 onboarding in 1.25 weeks.

---

## Curriculum Stages

| Stage | Clients | Dev Types | Max Roles/Project | Deadlines |
|---|---|---|---|---|
| 1 (easy) | 1 | 1 (backend only) | 1 | 8–14 weeks |
| 2 (medium) | 3 | 3 | 2 | 6–10 weeks |
| 3 (full) | 3+ | 5 | 3 | 4–10 weeks |

Set at startup via env var or hot-patch live:
```bash
# At startup
CURRICULUM_STAGE=2 uv run python -m uvicorn server.app:app --port 8000

# Live hot-patch (no restart needed)
curl -X PATCH http://localhost:8000/config/env -H "Content-Type: application/json" \
     -d '{"curriculum_stage": 2}'
```

---

## LLM Mode

```bash
# Stub (default — no API key, fast, deterministic-ish)
LLM_MODE=stub uv run python -m uvicorn server.app:app --port 8000

# Live (uses local Ollama / vLLM for rich semantic evaluations)
LLM_MODE=live OPENAI_API_BASE=http://localhost:11434/v1 \
    uv run python -m uvicorn server.app:app --port 8000
```

---

## File Structure

```
rl-recruits/
├── env/
│   ├── config.py            ← Config (env params) + TrainingConfig (training HPs)
│   ├── models.py            ← Candidate, Role, Project, Client dataclasses
│   ├── core.py              ← StaffingCore: all tool logic + world_tick()
│   ├── llm.py               ← LLMRouter: stub + live (Ollama/vLLM) implementations
│   └── simulation.py        ← World dynamics: arrivals, deadlines, patience, churn
├── server/
│   ├── staffing_environment.py  ← StaffingAgencyEnvironment(MCPEnvironment)
│   │                               Reward flow: _last_tool_reward pattern
│   └── app.py               ← create_app() + /config GET/PATCH endpoints
├── training/
│   ├── train_grpo.py        ← Entry point: parse_args + dispatch
│   ├── reinforce.py         ← REINFORCE-GRPO training loop
│   ├── rollout.py           ← rollout_full_episode() — live env interaction
│   ├── prompts.py           ← SYSTEM_PROMPT, TOOLS, parse_tool_call()
│   ├── policies.py          ← Heuristic baselines (random/greedy/optimal)
│   ├── dry_run.py           ← GPU-free validation simulator
│   └── metrics.py           ← Reward curve plots + JSON summaries
├── ui/
│   └── dashboard.py         ← Gradio dashboard (real cumulative_reward from server)
├── tests/
│   └── test_env.py          ← Environment unit tests
├── client.py                ← StaffingAgencyEnv(EnvClient) — sync + async
├── models.py                ← StaffingAction, StaffingObservation, StaffingState
├── openenv.yaml             ← OpenEnv manifest
├── pyproject.toml
└── README.md
```

---

## Will There Be Improvement Over 10 Episodes?

10 episodes is extremely short for a 52-week stateful environment. Here's what to expect realistically:

| Episodes | Expected Behaviour |
|---|---|
| 1–10 | Model learns basic tool-call syntax; rewards mostly negative (interview + onboarding costs) |
| 10–50 | Learns to call `advance_week`; starts seeing positive billing rewards |
| 50–200 | Learns to avoid benching; salary negotiation improves; project selection improves |
| 200+ | Systematic profit maximisation: demand-aware hiring, deadline management |

**To see faster improvement:**
1. Start at `curriculum_stage=1` (1 client, 1 role projects, long deadlines)
2. Use a smaller model (Qwen2.5-1.5B) for faster rollouts
3. Increase `num_episodes` to at least 50–100
4. Enable W&B to track the `env_reward` curve (distinct from `profit`)

## Making Training Faster

| Technique | How |
|---|---|
| Smaller model | `--model_name Qwen/Qwen2.5-1.5B-Instruct` |
| Shorter episodes | `CURRICULUM_STAGE=1` (fewer projects, faster sealing) |
| Fewer turns/week | `--max_turns_per_step 5` (forces faster decisions) |
| Parallel env instances | Deploy multiple server replicas on different ports |
| LoRA fine-tuning | Add `peft` + use `get_peft_model()` in reinforce.py |
| vLLM inference | Replace `model.generate()` in rollout.py with vLLM async API |

---

## Judging Criteria Alignment

| Criterion | How We Address It |
|---|---|
| **Environment Innovation (40%)** | Multi-actor (clients + candidates), sparse multi-role sealing, LLM-graded transitions, 52-step horizon, live config API |
| **Storytelling (30%)** | Clear CEO framing, economic tensions, rich tool descriptions |
| **Reward Improvement (20%)** | `--dry_run` shows random→greedy→optimal curves; REINFORCE with per-step env rewards |
| **Training Pipeline (10%)** | Custom REINFORCE loop with KL penalty, discounted returns, W&B logging |
