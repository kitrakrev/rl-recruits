# rl-recruits: Bugs & Issues Found in Training Pipeline

Audit date: 2026-03-08

---

## BUG 1 — CRITICAL: GRPO receives ZERO gradient signal

**Location:** `training/train_grpo.py` lines 1088–1092, 1179

**Problem:**
The Monte Carlo reward function assigns every completion in a GRPO group the **same** reward value (`final_profit / 1000`). TRL's `GRPOTrainer` generates `num_generations=8` completions per prompt, then computes group-relative advantages:

```
advantage_i = reward_i − mean(rewards_in_group)
```

Since **all 8 completions receive the same `mc_return`**, every advantage is exactly **0.0**. The policy gradient is zero. **The model learns nothing.**

```python
# Line 1179 — every step in the trajectory gets the same final_profit
mc_returns = [final_profit] * len(prompts)

# Line 1088 — reward function returns the same value for all completions
def reward_fn_mc_profit(completions, mc_return=None, **kwargs):
    if mc_return:
        return [float(r) / 1_000.0 for r in mc_return]  # all identical!
    return [0.0] * len(completions)
```

**Root cause:**
GRPO needs **variation** in rewards *within* a group to produce non-zero advantages. Since the reward function ignores the actual `completions` text and just returns the cached MC return, all completions are rewarded equally regardless of quality.

**Fix options:**

- **Option A (recommended): Online reward — execute each generated completion against the env.** For each of the 8 generated completions, parse the tool call, send it to the environment, and use the per-step `result.reward` from the server. This gives true differentiation: a good tool call (e.g. profitable match) gets high reward, a bad one (parse error, bad match) gets 0 or negative. Requires running the env during training (already available at localhost:8000).

- **Option B: Format + heuristic reward.** Don't call the env, but score each completion locally: +1.0 for valid tool-call syntax, +0.5 bonus for calling a contextually-appropriate tool (e.g. `hire_candidate` when pipeline has candidates), −1.0 for parse failure. This gives GRPO something to differentiate on, but the reward doesn't come from the env (violates the user's constraint).

- **Option C: Use per-step env rewards as MC return, not final profit.** During rollout, accumulate the per-step `result.reward` into a discounted return per step. Assign the *step-specific* return (not the flat final profit) as each step's MC return. This still doesn't help within a GRPO group (all 8 completions for the same prompt still get the same return), but at least different prompts (different weeks) get different returns — more variance across the dataset.

**Best fix: Option A.** It satisfies the constraint that "all rewards come from interacting with the agent."

---

## BUG 2 — UI dashboard uses fake reward (profit / 1000)

**Location:** `ui/dashboard.py` line 172

**Problem:**
```python
_history["reward"].append(a.get("current_profit", 0) / 1000)
```

The dashboard fabricates the "reward" metric by dividing profit by 1000, instead of using the actual `cumulative_reward` from the server. This means the Reward chart is always exactly `Profit / 1000` — the two charts show identical shapes and provide no new information.

**Fix:** Use the real cumulative_reward from the server's `_ctx` response.

---

## BUG 3 — W&B "reward" and "profit" are the same metric (scaled)

**Location:** `training/train_grpo.py` lines 1198–1221

**Problem:**
The logged `reward` comes from TRL's trainer log history, which reports the mean output of `reward_fn_mc_profit`. That function returns `final_profit / 1000`. So:

```
W&B "profit" = final_profit          (e.g. $50,000)
W&B "reward" = final_profit / 1000   (e.g. 50.0)
```

They are perfectly correlated. Having both metrics adds no diagnostic value — they always go up and down together. You can never distinguish "is the model getting better at tool use?" from "did profit happen to be higher this episode?"

**Fix:** Log the actual cumulative per-step reward from the environment rollout separately from profit. This shows the per-action reward signal (tool costs, margins, penalties) vs. the final financial outcome.

---

## BUG 4 — Per-step env rewards are completely ignored in training

**Location:** `training/train_grpo.py` lines 986–995

**Problem:**
During rollout, each `env.step()` returns a per-step `result.reward` that includes:
- Interview cost (−$500)
- Onboarding cost (−$2,000)
- Severance cost (−salary × 2 weeks)
- Speed bonus for fast sealing
- World tick: billing margins, bench burn, expiry penalties, churn penalties

This rich per-step signal is **only used for printing** (line 995). The training signal (line 1179) uses only `final_profit = revenue - costs`, which is a single scalar for the entire 52-week episode.

**Impact:** The per-step reward from the env carries important credit-assignment information (which specific action was good/bad). By collapsing everything to a single final profit, the model gets no signal about which of its ~500 actions in the episode actually mattered.

**Fix:** During rollout, accumulate per-step rewards and use them (e.g., as discounted returns or step-level MC returns) instead of assigning the same flat final_profit to every step.

---

## BUG 5 — TRL may not pass `mc_return` to reward function

**Location:** `training/train_grpo.py` lines 1088, 1180–1184

**Problem:**
The dataset contains a `mc_return` column. The reward function expects it as a keyword argument:

```python
def reward_fn_mc_profit(completions, mc_return=None, **kwargs):
```

Whether TRL's `GRPOTrainer` actually passes extra dataset columns (beyond `prompt`) to the reward function depends on the TRL version. In some versions, only `prompts` and `completions` are passed. If `mc_return` is not forwarded, the function returns `[0.0] * len(completions)` — meaning zero reward for everything.

**Fix:** Verify with the installed TRL version. If extra columns aren't forwarded, embed the MC return directly into the prompt text, or use a closure that captures the returns by prompt index.

---

## BUG 6 — Training uses stale rewards for fresh completions

**Location:** `training/train_grpo.py` lines 1187–1195

**Problem:**
TRL's `GRPOTrainer` takes the prompts from the dataset but generates **new** completions with the current model (that's the whole point of GRPO — on-policy generation). However, the reward for these new completions comes from `mc_return`, which was computed from the **old** rollout's completions. The new completions may call completely different tools, yet they receive the same reward as the original trajectory.

**Impact:** The reward signal is decorrelated from the actual completion quality. The model could generate `get_agency_state` 8 times in a row and still receive a reward of $50k/1000 if the original rollout was profitable.

**Fix:** This is the same root cause as Bug 1. The reward function must evaluate the **actual generated completion**, not return a cached value. Use online env interaction (Option A from Bug 1).

---

## ISSUE 7 — Server penalties are invisible but silently affect profit

**Location:** `server/staffing_environment.py` lines 143–158

**Problem:**
The server adds two behavioral penalties:
- Repeat-call penalty: −$100 for calling the same tool twice consecutively
- Passive-streak penalty: −$50/turn after 3+ consecutive GET-only calls

These penalties are added to `self.core.costs` (lines 146, 158), which means they **do** reduce `profit = revenue − costs`. But they are NOT visible in the per-step reward logged during training (they're added to `total_reward` in the server, but the training loop's print statement only shows `result.reward` and `final_profit`).

**Impact:** The agent might be penalized without the training logs showing why profit dropped. Debugging becomes harder.

**Note:** These penalties DO come from the environment server, so they satisfy the constraint "all rewards from interacting with the agent." They're just poorly surfaced in logging.

**Fix:** Log these penalties explicitly during training rollout so it's clear when they fire.

---

## ISSUE 8 — `advance_week` doesn't strip `reward` from tool result dict

**Location:** `server/staffing_environment.py` line 321–323

**Problem:**
Every other tool strips the `reward` key from the result dict before returning:
```python
if "reward" in res: res.pop("reward")
```

But `advance_week` returns the full dict including `reward`:
```python
def advance_week() -> dict:
    return env.core.tool_advance_week()  # includes "reward" key
```

This means the `reward` value from `world_tick()` is exposed to the agent in the tool result text (visible in the conversation). The agent could learn to game its reasoning based on seeing explicit reward numbers.

**Fix:** Strip `reward` from `advance_week`'s tool result dict, consistent with all other tools. The reward signal should flow through `obs.reward` (the Observation), not through the tool result text.

---

## Summary Table

| # | Severity | Issue | Impact |
|---|----------|-------|--------|
| 1 | **CRITICAL** | GRPO advantage is always 0 | Model learns nothing |
| 2 | **HIGH** | UI reward = profit/1000 | Dashboard shows fake metric |
| 3 | **HIGH** | W&B reward = profit/1000 | Redundant logging, no insight |
| 4 | **MEDIUM** | Per-step rewards unused | Poor credit assignment |
| 5 | **MEDIUM** | mc_return may not reach reward fn | Possible zero reward |
| 6 | **MEDIUM** | Stale rewards for fresh completions | Decorrelated training signal |
| 7 | **LOW** | Server penalties invisible in logs | Hard to debug |
| 8 | **LOW** | advance_week leaks reward in text | Agent sees reward numbers |
