# Training Guide — Staffing Agency RL Agent

> How the RL training loop, environment, reward system, and model interact.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Current Model](#2-current-model)
3. [Environment Economics](#3-environment-economics)
4. [The Action Pipeline](#4-the-action-pipeline)
5. [`advance_week` — The Core Loop](#5-advance_week--the-core-loop)
6. [Reward Function](#6-reward-function)
7. [State Injection & Pre-computed Matches](#7-state-injection--pre-computed-matches)
8. [Known Failure Modes](#8-known-failure-modes)
9. [Model Size Recommendations](#9-model-size-recommendations)
10. [Configuration Reference](#10-configuration-reference)
11. [Running Training](#11-running-training)

---

## 1. Architecture Overview

```
┌─────────────────┐     WebSocket/HTTP      ┌─────────────────────┐
│  Training Loop   │ ◀──────────────────────▶│  OpenEnv Server     │
│  (train_grpo.py) │   env.step(action)      │  (port 8001)        │
│                  │   → observation+reward   │                     │
│  ┌────────────┐  │                         │  StaffingAgency-    │
│  │ LLM (Qwen) │  │                         │  Environment        │
│  │ + LoRA     │  │                         │  (MCPEnvironment)   │
│  └────────────┘  │                         │                     │
│                  │                         │  ┌───────────────┐  │
│  rollout.py      │                         │  │ StaffingCore  │  │
│  - builds state  │                         │  │ - world_tick  │  │
│  - parses tools  │                         │  │ - 17 tools    │  │
│  - tracks reward │                         │  │ - candidates  │  │
│                  │                         │  │ - clients     │  │
└─────────────────┘                         │  └───────────────┘  │
                                            └─────────────────────┘
```

**Training method:** REINFORCE with per-step rewards (profit delta).  
**LoRA:** Rank 16, α=32, applied to all linear layers.  
**Episode:** 52 simulated business weeks. Up to 10 tool calls per week.

---

## 2. Current Model

| Setting | Value |
|---------|-------|
| **Model** | `Qwen/Qwen3-0.6B` (600M params) |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **Context window** | 2048 tokens (left-truncated) |
| **Generation** | max 512 new tokens, temp=0.7, top_p=0.8, top_k=20 |
| **Thinking mode** | Disabled (`enable_thinking=False`) |

Defined in `training/config.py → TrainingConfig`.

The 0.6B model is intended for **smoke-testing the pipeline**. It cannot reason about type matching, margins, or multi-step planning. See [Model Size Recommendations](#9-model-size-recommendations) for production alternatives.

---

## 3. Environment Economics

All economics are defined in `env/config.py → Config`.

### Revenue (the only way to earn money)
- A **placed** candidate generates `margin_weekly = bill_rate_weekly − salary_weekly` per week
- Bill rates are set per role: $130K–$300K+/yr ($2,500–$5,800/wk)
- Margin can be **negative** if salary > bill_rate

### Costs
| Action | Cost |
|--------|------|
| `interview_candidate` | $500 per interview |
| `hire_candidate` | $2,000 onboarding fee |
| Bench burn (hired, not placed) | `salary_weekly` per week (varies by candidate) |
| `let_go_candidate` | 2× weekly salary severance |
| Expired project penalty | `unfilled_headcount × ($85K/52) × 1.25 × deadline_remaining` |
| Client churn (satisfaction < 0.3) | $50,000 LTV penalty |

### Salary Generation
```
annual_salary = base_salary[seniority] × role_multiplier[dev_type] × skill_modifier × rng_variance
```
- Base salaries: junior=$75K, mid=$110K, senior=$150K
- Role multipliers: frontend=1.0, backend=1.05, fullstack=1.1, devops=1.15, ml_engineer=1.3
- Skill modifier: `0.8 + skill_score × 0.4` (maps [0.3, 1.0] → [0.92, 1.20])
- RNG variance: uniform [0.80, 1.20]

### Bill Rate Generation
```
bill_rate = base_salary[seniority] × role_multiplier[dev_type] × scarcity_premium × client_premium
```
- Scarcity: ml_engineer=1.15, devops=1.10, backend=1.05, fullstack=1.02, frontend=1.0
- Client premium: uniform [1.30, 2.00]

**Key insight:** Bill rates are always 1.3×–2.0× the base, while salaries include a ±20% variance and skill modifier. Cheap juniors on high-bill-rate roles = maximum margin.

---

## 4. The Action Pipeline

Candidates must progress through a strict pipeline:

```
available (market)
    │
    ▼  interview_candidate($500)
in_pipeline
    │
    ▼  hire_candidate($2,000)
hired (on bench — salary burns every week!)
    │
    ▼  match_candidate_to_project(candidate_id, project_id, role_id)
placed (generating margin every week)
```

**Critical rules for `match_candidate_to_project`:**
1. Candidate must have status `"hired"` (not `"in_pipeline"` or `"available"`)
2. Candidate `developer_type` must be in the role's **adjacency set** (see below)
3. Candidate `skill_score` must be ≥ role's `min_skill_score`
4. Candidate `seniority_level` must be ≥ role's `seniority` (senior ≥ mid ≥ junior)

### Type Adjacency Matrix

| Candidate Type | Can Fill Roles |
|----------------|---------------|
| backend | backend, fullstack, ml_engineer |
| frontend | frontend, fullstack |
| fullstack | fullstack, backend, frontend |
| ml_engineer | ml_engineer, backend |
| devops | devops only |

A `frontend` candidate **cannot** fill a `backend` role. A `fullstack` candidate **can** fill backend, frontend, or fullstack roles.

---

## 5. `advance_week` — The Core Loop

When the agent calls `advance_week`, `world_tick()` executes 8 substeps in order:

### Step 1: Billing (revenue)
```python
for each placed candidate:
    cash += bill_rate_weekly
    cash -= salary_weekly
    reward += margin_weekly  # (bill_rate - salary)
```
**This is the only way to earn money.** No placements → advance_week returns reward=0.

### Step 2: Bench Burn
```python
for each hired (not placed) candidate:
    cash -= salary_weekly
    reward -= salary_weekly
```
Every unplaced hire drains their full salary every week.

### Step 3: Project Arrivals
New projects arrive stochastically (Poisson λ=0.5 per client per week).

### Step 4: Deadline Expiry
Project deadlines decrement by 1. Expired unfilled projects incur:
```
penalty = unfilled_headcount × ($85K/52) × 1.25 × deadline_remaining
```

### Step 5: Contract Completions
Placed candidates whose `contract_weeks_left` reaches 0 return to bench (status → `"hired"`).

### Step 6: Candidate Patience
Candidates in pipeline too long may leave the market entirely.

### Step 7: Client Churn
If `satisfaction_score < 0.3`, the client churns → $50,000 LTV penalty.

### Step 8: Market Replenishment
New candidates appear if market pool is below `market_pool_size` (20).

---

## 6. Reward Function

Rewards are **per-step profit deltas** — the change in `revenue - costs` from one action to the next.

| Source | Reward |
|--------|--------|
| Successful placement (`match_candidate_to_project`) | `max(margin_weekly, 0)` (immediate bonus) |
| `advance_week` — placed candidate billing | `+margin_weekly` per placed candidate |
| `advance_week` — bench burn | `-salary_weekly` per benched candidate |
| `interview_candidate` | `-$500` |
| `hire_candidate` | `-$2,000` |
| Failed action tool call | `-$100` (invalid_action_penalty) |
| Project expiry | `-unfilled × $2,043 × deadline_remaining` |
| Client churn | `-$50,000` |
| Speed bonus (sealed project within 2 weeks) | `+margin × 0.1` |

**No reward for GET/passive tools** (find_candidate, find_available_projects, get_*).

---

## 7. State Injection & Pre-computed Matches

The small model can't figure out type-matching on its own. The `_build_actionable_state()` function in `training/rollout.py` pre-computes valid matches and presents them as copy-pasteable commands:

```
[Week 7 of 52]

★ READY-TO-EXECUTE MATCHES (call these NOW, then advance_week):
  → match_candidate_to_project(candidate_id="C-BA-e6581e1b",
    project_id="P-CL-abc123", role_id="R-P-CL-abc123-0")  margin=$1,200/wk

★ HIRE THEN MATCH (call hire_candidate first, then match):
  → hire_candidate(candidate_id="C-ML-4e6cafdb") then
    match_candidate_to_project(...)  est_margin=$800/wk

★ INTERVIEW THESE (matching types for open roles):
  → interview_candidate(candidate_id="C-BA-f1234567")
    backend mid  salary_exp=$1,500/wk

DO NOW: Execute the ★ matches above, then call advance_week.
```

This is injected as a `"user"` message at the start of each week. The function:
1. Queries open projects and unfilled roles
2. Queries hired/pipeline/placed candidates and market candidates
3. Uses the adjacency matrix + seniority rules + margin check to find valid pairs
4. Presents them in priority order: ready matches > hire-then-match > interview candidates

---

## 8. Known Failure Modes

### 8.1 Model calls `advance_week` immediately every week
**Symptom:** Cash stays at $50,000, reward=0, for 17+ weeks.  
**Cause:** Model hasn't learned to take actions before advancing.  
**Mitigation:** The state injection ends with "DO NOW: ..." directive.

### 8.2 Model spam-repeats `find_candidate` or `match_candidate_to_project`
**Symptom:** 10 identical calls per week, all returning rew=0 or rew=-100.  
**Cause:** Model is too small to understand tool-call results. Gets stuck in a loop.  
**Mitigation:** Auto-advance after 10 turns. Hard-reset after 3 parse failures.

### 8.3 `match_candidate_to_project` always fails (ΔP=0, rew=-100)
**Symptom:** Model does interview→hire→match but match returns `success: False`.  
**Cause:** Type mismatch (frontend candidate → backend role), skill too low, or candidate not yet hired.  
**Mitigation:** Pre-computed matches in state injection. Invalid action penalty (-$100).

### 8.4 Cash decreasing despite placements
**Symptom:** `advance_week` shows negative ΔP even though some candidates are placed.  
**Cause:** Bench burn from multiple unplaced candidates exceeds billing from placed ones.  
**Mitigation:** Agent needs to learn to only hire when it can immediately place.

### 8.5 Parse failures (model echoes state text instead of tool call)
**Symptom:** `[parse-fail]` with raw text containing the state injection text.  
**Cause:** Small model confuses instructions with actions. The `★` and `→` characters look like things to output.  
**Mitigation:** Soft nudge after 1 failure, hard-reset conversation after 3.

---

## 9. Model Size Recommendations

| Model | Params | Tool-call Quality | Type Reasoning | RL Learning | Speed |
|-------|--------|-------------------|----------------|-------------|-------|
| **Qwen3-0.6B** (current) | 600M | Poor — frequent parse failures, empty `<think>` blocks | Cannot reason about types | Minimal — needs pre-computed everything | ~1s/step |
| **Qwen3-1.7B** | 1.7B | Decent — fewer parse failures | Still needs hand-holding | Can learn basic patterns | ~3s/step |
| **Qwen3-4B** | 4B | Good — reliable tool-call formatting | Can follow explicit rules | **Recommended minimum for RL** | ~6s/step |
| **Qwen3-8B** | 8B | Excellent — production-grade | Can reason about margins/types | Full RL training viable | ~12s/step |

### Recommendation
- **Debugging/pipeline testing:** Keep Qwen3-0.6B
- **Actual RL training:** Use **Qwen3-4B** minimum (change `model_name` in `training/config.py`)
- **Production/best results:** Use **Qwen3-8B** (as noted in the config comments)

To switch models:
```python
# training/config.py
model_name: str = "Qwen/Qwen3-4B"  # or "Qwen/Qwen3-8B"
```
Or via CLI:
```bash
uv run python training/train_grpo.py --model_name Qwen/Qwen3-4B --num_episodes 50
```

---

## 10. Configuration Reference

### Environment Config (`env/config.py → Config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episode_steps` | 52 | Weeks per episode |
| `seed_capital` | $50,000 | Starting cash |
| `curriculum_stage` | 3 | 1=simple, 2=medium, 3=full |
| `num_clients` | 3 | Number of client companies |
| `market_pool_size` | 20 | Max candidates in market |
| `contract_duration` | 26 | Weeks per placement contract |
| `onboarding_cost` | $2,000 | Cost per hire |
| `cost_per_interview` | $500 | Cost per interview |
| `invalid_action_penalty` | -$100 | Penalty for failed action calls |
| `churn_threshold` | 0.3 | Satisfaction below → client churns |
| `client_ltv_estimate` | $50,000 | Churn penalty amount |

### Training Config (`training/config.py → TrainingConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-0.6B` | HuggingFace model ID |
| `lora_rank` | 16 | LoRA adapter rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `learning_rate` | 5e-6 | Optimiser LR |
| `gamma` | 0.99 | Discount factor |
| `kl_coeff` | 0.05 | KL penalty weight |
| `max_turns_per_step` | 10 | Max tool calls per week |
| `max_prompt_len` | 2048 | Left-truncation limit |
| `max_new_tokens` | 512 | Generation budget per turn |
| `num_episodes` | 200 | Total training episodes |

---

## 11. Running Training

### 1. Start the environment server
```bash
cd /root/rl-recruits
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8001 &
```

### 2. Verify server health
```bash
curl -s http://localhost:8001/health
```

### 3. Run training
```bash
# Smoke test (10 episodes, 0.6B model)
uv run python training/train_grpo.py --num_episodes 10 --env_url http://localhost:8001

# Production run (4B model, 200 episodes)
uv run python training/train_grpo.py \
    --model_name Qwen/Qwen3-4B \
    --num_episodes 200 \
    --env_url http://localhost:8001

# Dry run (no GPU, simulated rewards)
uv run python training/train_grpo.py --dry_run --num_episodes 90
```

### 4. Monitor via logs
The training loop prints per-action traces:
```
W07T01 interview_candidate       - cash=$  42,795  ΔP=    -500  rew= -500.00
W07T02 hire_candidate            - cash=$  40,795  ΔP=  -2,000  rew=-2000.00
W07T03 match_candidate_to_project = cash=$  40,795  ΔP=      +0  rew= +742.31  ← SUCCESS!
W07T04 advance_week              + cash=$  41,538  ΔP=    +742  rew= +742.31
```

| Symbol | Meaning |
|--------|---------|
| `+` | Profit increased (cash went up) |
| `-` | Profit decreased (cost incurred) |
| `=` | No profit change (GET call or failed action) |
| `[parse-fail]` | Model output couldn't be parsed as a tool call |
| `[auto-advance]` | Max turns reached, week auto-advanced |

### 5. Checkpoints
Saved to `training/checkpoints/` with LoRA weights, tokenizer, and training state.
