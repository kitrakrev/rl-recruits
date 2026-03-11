"""
Episode rollout functions.

rollout_full_episode  — live 52-week rollout using model.generate().
                        Returns per-step rewards for the REINFORCE update.

rollout_episode       — OpenEnv-compatible rollout that accepts any callable
                        generate function (useful for testing / integration).
"""
from __future__ import annotations

import json
import re
import random
from typing import TYPE_CHECKING, Callable

from training.prompts import SYSTEM_PROMPT, TOOLS, KNOWN_TOOLS, parse_tool_call

# Alias for local use — sourced from prompts so always in sync with TOOLS
_KNOWN_TOOLS: frozenset[str] = KNOWN_TOOLS

if TYPE_CHECKING:
    from training.config import TrainingConfig


def _build_actionable_state(env, week: int) -> str:
    """Build a compact, actionable state string with concrete IDs the model can use.

    The key insight: the small model can't figure out type-matching on its own.
    We pre-compute valid (candidate, role) pairs and present them as ready-to-use
    commands so the model just needs to pick one and execute the pipeline.
    """
    from models import StaffingAction

    # Type adjacency (must match env/config.py)
    ADJACENCY = {
        "backend":     {"backend", "fullstack", "ml_engineer"},
        "frontend":    {"frontend", "fullstack"},
        "fullstack":   {"fullstack", "backend", "frontend"},
        "ml_engineer": {"ml_engineer", "backend"},
        "devops":      {"devops"},
    }
    SEN_ORDER = {"junior": 0, "mid": 1, "senior": 2}

    lines: list[str] = [f"[Week {week} of 52]"]

    # --- Open projects with unfilled roles ---
    try:
        r = env.step(StaffingAction(tool="find_available_projects", params={}))
        projects = (r.observation.tool_result or {}).get("projects", [])
    except Exception:
        projects = []

    # Flatten to list of unfilled roles with their project context
    open_roles: list[dict] = []
    for p in projects[:6]:
        pid = p["project_id"]
        deadline = p.get("deadline_remaining", "?")
        for role in p.get("roles", []):
            if role.get("is_filled"):
                continue
            open_roles.append({
                "project_id": pid,
                "role_id": role["role_id"],
                "developer_type": role.get("developer_type", "?"),
                "seniority": role.get("seniority", "junior"),
                "bill_rate_weekly": role.get("bill_rate_weekly", 0),
                "deadline": deadline,
            })

    # --- Your candidates ---
    try:
        r = env.step(StaffingAction(tool="get_candidate_state", params={}))
        cstate = r.observation.tool_result or {}
    except Exception:
        cstate = {}

    all_cands = cstate.get("candidates", [])
    hired = [c for c in all_cands if c.get("status") == "hired"]
    pipeline = [c for c in all_cands if c.get("status") == "in_pipeline"]
    placed = [c for c in all_cands if c.get("status") == "placed"]

    # --- Market candidates ---
    try:
        r = env.step(StaffingAction(tool="find_candidate", params={}))
        market = (r.observation.tool_result or {}).get("candidates", [])
    except Exception:
        market = []

    # ================================================================
    # SECTION 1: READY-TO-EXECUTE MATCHES (hired candidate ↔ open role)
    # ================================================================
    ready_matches: list[str] = []
    used_hired: set[str] = set()
    used_roles: set[str] = set()

    for role in open_roles:
        for c in hired:
            cid = c["id"]
            if cid in used_hired:
                continue
            ctype = c.get("developer_type", "?")
            csen = c.get("seniority_level", "junior")
            rtype = role["developer_type"]
            rsen = role["seniority"]
            # Check adjacency
            if rtype not in ADJACENCY.get(ctype, set()):
                continue
            # Check seniority
            if SEN_ORDER.get(csen, 0) < SEN_ORDER.get(rsen, 0):
                continue
            salary = c.get("salary_weekly", 0)
            bill = role["bill_rate_weekly"]
            margin = bill - salary
            if margin <= 0:
                continue  # would lose money
            ready_matches.append(
                f'  → match_candidate_to_project(candidate_id="{cid}", '
                f'project_id="{role["project_id"]}", role_id="{role["role_id"]}")'
                f"  margin=${margin:,.0f}/wk"
            )
            used_hired.add(cid)
            used_roles.add(role["role_id"])
            break  # one candidate per role

    if ready_matches:
        lines.append("\n★ READY-TO-EXECUTE MATCHES (call these NOW, then advance_week):")
        lines.extend(ready_matches)

    # ================================================================
    # SECTION 2: PIPELINE CANDIDATES → hire then match
    # ================================================================
    pipeline_actions: list[str] = []
    for c in pipeline[:4]:
        cid = c["id"]
        ctype = c.get("developer_type", "?")
        csen = c.get("seniority_level", "junior")
        salary_exp = c.get("salary_expectation", 0)
        # Find a compatible open role
        for role in open_roles:
            if role["role_id"] in used_roles:
                continue
            rtype = role["developer_type"]
            rsen = role["seniority"]
            if rtype not in ADJACENCY.get(ctype, set()):
                continue
            if SEN_ORDER.get(csen, 0) < SEN_ORDER.get(rsen, 0):
                continue
            margin = role["bill_rate_weekly"] - salary_exp
            if margin <= 0:
                continue
            pipeline_actions.append(
                f'  → hire_candidate(candidate_id="{cid}") then '
                f'match_candidate_to_project(candidate_id="{cid}", '
                f'project_id="{role["project_id"]}", role_id="{role["role_id"]}")'
                f"  est_margin=${margin:,.0f}/wk"
            )
            break

    if pipeline_actions:
        lines.append("\n★ HIRE THEN MATCH (call hire_candidate first, then match):")
        lines.extend(pipeline_actions)

    # ================================================================
    # SECTION 3: MARKET → interview → hire → match
    # ================================================================
    # Find market candidates whose type matches an open role
    needed_types: set[str] = set()
    for role in open_roles:
        if role["role_id"] not in used_roles:
            needed_types.add(role["developer_type"])
            # Also add adjacent types that can fill this role
            for ctype, adj in ADJACENCY.items():
                if role["developer_type"] in adj:
                    needed_types.add(ctype)

    market_actions: list[str] = []
    for c in market[:6]:
        ctype = c.get("developer_type", "?")
        if ctype not in needed_types and needed_types:
            continue
        cid = c["id"]
        csen = c.get("seniority_level", "junior")
        salary_exp = c.get("salary_expectation", 0)
        market_actions.append(
            f'  → interview_candidate(candidate_id="{cid}")  '
            f"{ctype} {csen}  salary_exp=${salary_exp:,.0f}/wk"
        )
        if len(market_actions) >= 3:
            break

    if market_actions:
        lines.append("\n★ INTERVIEW THESE (matching types for open roles):")
        lines.extend(market_actions)
    elif not ready_matches and not pipeline_actions:
        # Nothing useful to do — just show any market candidate
        if market:
            c = market[0]
            lines.append(
                f'\nNo matching candidates. Interview anyway: '
                f'interview_candidate(candidate_id="{c["id"]}")'
            )

    # ================================================================
    # SECTION 4: Summary + what to do
    # ================================================================
    if placed:
        lines.append(f"\nPLACED (earning revenue): {len(placed)} candidates")

    if not open_roles:
        lines.append("\nNo open projects. Call advance_week to get new projects.")
    elif ready_matches:
        lines.append("\nDO NOW: Execute the ★ matches above, then call advance_week.")
    elif pipeline_actions:
        lines.append("\nDO NOW: Hire the pipeline candidates above, match them, then advance_week.")
    elif market_actions:
        lines.append("\nDO NOW: Interview a candidate above, hire, match to a role, then advance_week.")
    else:
        lines.append("\nNo viable candidates for open roles. Call advance_week.")

    return "\n".join(lines)


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from generated text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Full 52-week rollout (used by the REINFORCE training loop)
# ---------------------------------------------------------------------------

def rollout_full_episode(
    env,
    model,
    tokenizer,
    seed: int,
    max_turns_per_step: int = 10,
    rng: random.Random | None = None,
    cfg: "TrainingConfig | None" = None,
) -> tuple[list[str], list[str], list[float], float, float]:
    """
    Run one complete 52-week episode with the current model.

    At each turn the model generates a tool call, which is executed against the
    live environment server. The env returns a per-step reward that encodes
    real business consequences (interview costs, onboarding, billing margins,
    expiry penalties, client churn, etc.). All rewards come from env.step() —
    no fabricated rewards.

    Parameters
    ----------
    env         : StaffingAgencyEnv context manager (already opened)
    model       : HuggingFace AutoModelForCausalLM (may be a PeftModel)
    tokenizer   : matching AutoTokenizer
    seed        : episode seed (controls env stochasticity)
    max_turns_per_step : max tool calls per week before auto-advancing
    rng         : optional Random instance for tool-call IDs; creates one if None
    cfg         : TrainingConfig for generation / tokenisation settings

    Returns
    -------
    (prompts, completions, step_rewards, final_profit, cumulative_env_reward)
    """
    import torch

    from models import StaffingAction  # project root

    if rng is None:
        rng = random.Random(seed)

    # Resolve generation settings from cfg or use safe defaults
    max_prompt_len = cfg.max_prompt_len if cfg else 2048
    max_new_tokens = cfg.max_new_tokens if cfg else 512
    temperature    = cfg.temperature    if cfg else 0.7
    top_p          = cfg.top_p          if cfg else 0.8
    top_k          = cfg.top_k          if cfg else 20

    # Left-truncate so the generation prompt always stays at the end of the
    # context window.  Right-truncation would cut the "<|im_start|>assistant"
    # suffix, causing the model to complete a half-open tool_call from history.
    tokenizer.truncation_side = "left"

    env.reset(seed=seed)
    prompts_out: list[str]    = []
    completions_out: list[str] = []
    step_rewards: list[float]  = []
    cumulative_env_reward: float = 0.0
    final_profit = 0.0

    # Fetch episode length from the env so week labels are accurate
    episode_steps = 52  # default
    try:
        r = env.step(StaffingAction(tool="get_agency_state", params={}))
        episode_steps = int((r.observation.tool_result or {}).get("episode_steps", 52))
    except Exception:
        pass

    # Seed Week 1: give the model real IDs so it can act immediately.
<<<<<<< HEAD
    state_text = _build_actionable_state(env, 1)
    final_profit = float(
        env.step(StaffingAction(tool="get_financial_summary", params={}))
        .observation.profit or 0.0
    )
=======
    init_state: dict = {}
    try:
        r = env.step(StaffingAction(tool="get_financial_summary", params={}))
        init_state["financials"] = r.observation.tool_result or {}
        final_profit = float(r.observation.profit or 0.0)
    except Exception:
        pass
    try:
        r = env.step(StaffingAction(tool="get_candidate_state", params={}))
        init_state["candidates"] = r.observation.tool_result or {}
    except Exception:
        pass
    try:
        r = env.step(StaffingAction(tool="find_available_projects", params={}))
        raw = r.observation.tool_result or {}
        # Compact project list: show only what the model needs to plan hiring
        init_state["open_projects"] = [
            {"project_id": p.get("project_id"), "deadline_week": p.get("deadline_week"),
             "roles": [{"role_id": ro.get("role_id"), "type": ro.get("developer_type"),
                        "min_skill": ro.get("min_skill_score"), "bill_rate": ro.get("bill_rate_weekly")}
                       for ro in p.get("roles", []) if not ro.get("is_filled")]}
            for p in raw.get("projects", [])[:6]
        ]
    except Exception:
        pass
>>>>>>> origin/main

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
<<<<<<< HEAD
        {"role": "user", "content": state_text},
=======
        {"role": "user", "content": (
            f"Week 1 of {episode_steps}. Current state:\n{init_str}\n\n"
            "Take business actions. Find candidates, interview them, hire the best ones, "
            "confirm projects, and match candidates to project roles to generate profit. "
            "Call advance_week when done."
        )},
>>>>>>> origin/main
    ]

    print(f"\n      --- Rollout Phase (Seed {seed}) ---")

<<<<<<< HEAD
    prev_profit = final_profit
    episode_done = False
    for week in range(1, 53):
=======
    episode_done = False
    prev_profit  = final_profit
    # Use a high upper-bound; real termination comes from env `done` flag
    for week in range(1, 200):
>>>>>>> origin/main
        if episode_done:
            break
        week_advanced = False
        parse_fail_streak = 0

        for turn in range(1, max_turns_per_step + 1):
            # ---- Render prompt ------------------------------------------------
            try:
                prompt_str = tokenizer.apply_chat_template(
                    conversation,
                    tools=TOOLS,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Qwen3: disable <think> blocks for tool calls
                )
            except TypeError:
                # Older model / tokenizer that doesn't support enable_thinking
                prompt_str = tokenizer.apply_chat_template(
                    conversation,
                    tools=TOOLS,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            if week == 1 and turn == 1:
                print(f"  [prompt sample] ...{prompt_str[-300:]}\n")

            # ---- Tokenise (left-truncate) ------------------------------------
            input_ids = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            ).to(model.device)

            # ---- Generate ----------------------------------------------------
            with torch.no_grad():
                out_ids = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Only decode the newly generated tokens — not the prompt.
            # Parsing the full sequence caused the model to "re-trigger" old
            # tool calls that were already in the conversation history.
            completion_ids = out_ids[0][input_ids.input_ids.shape[-1]:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Strip any Qwen3 thinking blocks that leaked through despite enable_thinking=False
            completion_text = _strip_think(completion_text)

            parsed = parse_tool_call(completion_text)

            reward = 0.0
            final_cash = 0.0
            tool_result_text = "No tool parsed — no action taken."

            if parsed:
                parse_fail_streak = 0
                tool_name, params = parsed

                # Catch hallucinated tool names before they silently no-op
                if tool_name not in _KNOWN_TOOLS:
                    # Find closest real tool name
                    closest = min(_KNOWN_TOOLS, key=lambda t: abs(len(t) - len(tool_name)))
                    tool_result_text = (
                        f"ERROR: '{tool_name}' is not a valid tool name. "
                        f"Did you mean '{closest}'? "
                        f"Valid tools: {sorted(_KNOWN_TOOLS)}"
                    )
                    status = "="
                    print(f"  W{week:02d}T{turn:02d} [BAD TOOL: '{tool_name}' → did you mean '{closest}'?]")
                    # Feed correction back and skip the env.step
                    parsed_for_conv = parsed
                    parsed = None  # treat as parse fail so correction goes into conversation
                    parse_fail_streak = 0
                    conversation.append({"role": "assistant", "content": completion_text})
                    conversation.append({"role": "user", "content": tool_result_text})
                    prompts_out.append(prompt_str)
                    completions_out.append(completion_text)
                    step_rewards.append(0.0)
                    continue

                try:
                    result = env.step(StaffingAction(tool=tool_name, params=params))
                    reward = float(result.reward or 0.0)
                    final_profit = float(result.observation.profit or 0.0)
                    final_cash = float(result.observation.cash or 0.0)
                    tr = result.observation.tool_result or {}
                    profit_delta = final_profit - prev_profit
                    prev_profit = final_profit
                    if getattr(result.observation, "done", False):
                        episode_done = True

                    ok = tr.get("success", True)

                    # Build a clear, instructive tool response for the model.
                    # For failures: surface the enriched reason+fix from the server.
                    # For success: show key fields + any REQUIRED_NEXT_STEP prominently.
                    if not ok:
                        err    = tr.get("error", "unknown error")
                        reason = tr.get("reason", "")
                        fix    = tr.get("fix", "")
                        parts  = [f"FAILED: {err}"]
                        if reason:
                            parts.append(f"REASON: {reason}")
                        # Surface any useful context (bench, market IDs, etc.)
                        for key in ("your_bench", "your_hired_candidates",
                                    "current_market_ids", "available_in_market",
                                    "open_projects", "roles_matching_your_candidate"):
                            if key in tr:
                                parts.append(f"{key.upper()}: {json.dumps(tr[key])[:200]}")
                        if fix:
                            parts.append(f"FIX: {fix}")
                        if not reason and not fix:
                            parts.append("Check IDs and try again using the ACTION GUIDE.")
                        tool_result_text = "\n".join(parts)
                    else:
                        # Success — build a compact summary and always surface NEXT_STEP
                        core_fields = {k: v for k, v in tr.items()
                                       if k not in ("success",) and not k.startswith("_")}
                        tool_result_text = json.dumps(core_fields, indent=2)[:500]
                        # Pull REQUIRED_NEXT_STEP / next_step to the top so model sees it
                        for ns_key in ("REQUIRED_NEXT_STEP", "next_step", "roles_to_fill"):
                            if ns_key in tr:
                                tool_result_text = (
                                    f"⚠ NEXT: {tr[ns_key]}\n\n" + tool_result_text
                                )

                    status = "+" if profit_delta > 0 else ("-" if profit_delta < 0 else "=")
                    result_hint = ""
                    if not ok:
                        result_hint = f"  !! FAIL: {tr.get('error', '')[:80]}"
                    elif tool_name == "match_candidate_to_project":
                        result_hint = f"  (score={tr.get('match_score','?')} sealed={tr.get('project_sealed','?')})"
                    print(
                        f"  W{week:02d}T{turn:02d} {tool_name:25s} "
                        f"{status} cash=${final_cash:>8,.0f}  ΔP={profit_delta:>+8,.0f}  rew={reward:>+8.2f}"
                        f"{result_hint}"
                    )

                    if tool_name == "advance_week":
                        week_advanced = True
                    if result.done:
                        episode_done = True
                except Exception as e:
                    tool_result_text = f"Error: {e}"
                    print(f"  W{week:02d}T{turn:02d} {tool_name:25s} ! ERROR: {e}")
            else:
                parse_fail_streak += 1
                clean = completion_text.strip().replace("\n", " ")
                print(f"  W{week:02d}T{turn:02d} [parse-fail #{parse_fail_streak}]"
                      f"              = cash=${final_cash:>8,.0f}  raw: {clean[:100]}")

            prompts_out.append(prompt_str)
            completions_out.append(completion_text)
            step_rewards.append(reward)
            cumulative_env_reward += reward

            if week_advanced or episode_done:
                break

            # ---- Build next conversation turn --------------------------------
            tool_call_id = f"call_{rng.getrandbits(32):08x}"

            if parsed:
                # Standard: assistant made a valid tool call
                conversation.append({
                    "role": "assistant",
                    "content": None,  # avoid double-rendering by the chat template
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": parsed[0],
                            "arguments": json.dumps(parsed[1]),
                        },
                    }],
                })
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": parsed[0],
                    "content": tool_result_text,
                })
            elif parse_fail_streak >= 3:
                # Hard-reset: accumulated broken context makes things worse.
                # Start fresh with system prompt + a strict minimal prompt.
                print(f"  W{week:02d}   [HARD-RESET after {parse_fail_streak} parse fails]")
                parse_fail_streak = 0
                conversation = [
                    conversation[0],  # keep system prompt
                    {
                        "role": "user",
                        "content": (
                            f"[Week {week} of {episode_steps}] Your previous responses could not be parsed. "
                            "You MUST respond with a tool call and NOTHING else. Example:\n"
                            '<tool_call>\n{"name": "find_candidate", "arguments": {"developer_type": "backend"}}\n</tool_call>\n'
                            "Available tools: find_candidate, interview_candidate, hire_candidate, "
                            "negotiate_salary, match_candidate_to_project, get_client_state, "
                            "get_candidate_state, advance_week. Make a tool call now."
                        ),
                    },
                ]
            else:
                # Soft nudge: append the broken output and ask to retry
                conversation.append({"role": "assistant", "content": completion_text})
                conversation.append({
                    "role": "user",
                    "content": (
                        "No tool call detected. You MUST respond with ONLY a tool call, no prose. "
                        "Use this exact format:\n"
                        '<tool_call>\n{"name": "find_candidate", "arguments": {"developer_type": "backend"}}\n</tool_call>'
                    ),
                })

        # Auto-advance if model never called advance_week.
        # This is a safety valve — NOT a model decision.  Zero reward so
        # the model can't earn billing revenue by wasting all 10 turns.
        if not week_advanced:
            print(f"  W{week:02d}   [auto-advance — max turns reached]")
            try:
                result = env.step(StaffingAction(tool="advance_week", params={}))
                final_profit = float(result.observation.profit or 0.0)
<<<<<<< HEAD
                prev_profit = final_profit  # sync so next week's ΔP is clean
                if result.done:
                    episode_done = True
                # Reward is intentionally 0 — auto-advance is not a model action.
                step_rewards.append(0.0)
                prompts_out.append("")        # placeholder — no model output for auto-advance
=======
                auto_reward = float(result.reward or 0.0)
                step_rewards.append(auto_reward)
                cumulative_env_reward += auto_reward
                prompts_out.append("")
>>>>>>> origin/main
                completions_out.append("")
                if getattr(result.observation, "done", False):
                    episode_done = True
            except Exception:
                pass

        if episode_done:
<<<<<<< HEAD
=======
            print(f"  W{week:02d}   [episode DONE — env signalled termination]")
>>>>>>> origin/main
            break

        # Trim history and inject fresh state for next week
        system_msg = conversation[0]
        recent = conversation[-6:] if len(conversation) > 7 else conversation[1:]

<<<<<<< HEAD
        state_text = _build_actionable_state(env, week + 1)
        conversation = [system_msg] + recent + [{
            "role": "user",
            "content": state_text,
=======
        cand_state: dict = {}
        client_state: dict = {}
        try:
            r = env.step(StaffingAction(tool="get_candidate_state", params={}))
            cand_state = r.observation.tool_result or {}
        except Exception:
            pass
        try:
            r = env.step(StaffingAction(tool="get_client_state", params={}))
            client_state = r.observation.tool_result or {}
        except Exception:
            pass

        # ── Candidate roster split by status ─────────────────────────────────
        all_cands = cand_state.get("candidates_pool", [])
        # MATCHABLE: on bench, ready to be placed on a project
        bench    = [c for c in all_cands if c.get("status") == "hired"]
        # ALREADY PLACED: on a project — do NOT try to match or let_go these
        placed   = [c for c in all_cands if c.get("status") == "placed"]
        # INTERVIEWING: interviewed but not yet hired — call hire_candidate next
        pipeline = [c for c in all_cands if c.get("status") == "in_pipeline"]

        # ── Open roles across all non-sealed projects ─────────────────────
        open_roles = []
        for cl in client_state.get("clients", []):
            for proj in cl.get("projects", []):
                if proj.get("fill_status") == "SEALED":
                    continue
                for role in proj.get("roles", []):
                    if not role.get("is_filled", False):
                        open_roles.append({
                            "project_id": proj["project_id"],
                            "role_id":    role["role_id"],
                            "type":       role.get("developer_type", "?"),
                            "min_skill":  role.get("min_skill_score", 0),
                        })

        # ── Valid bench→role matches (type-compatible) ────────────────────
        valid_matches = []
        for c in bench:
            c_type  = c.get("developer_type", "")
            c_skill = c.get("skill_score", 0.0)
            for role in open_roles:
                if role["type"] == c_type:
                    valid_matches.append({
                        "candidate_id": c["id"],
                        "project_id":   role["project_id"],
                        "role_id":      role["role_id"],
                        "type":         c_type,
                    })

        lines = ["\n\n═══ ACTION GUIDE (use EXACT IDs — do not invent any) ═══"]

        lines.append(f"\nBENCH (status=hired, ready to match) — {len(bench)} candidates:")
        if bench:
            for c in bench[:6]:
                lines.append(f"  {c['id']:6s} | {c.get('developer_type','?'):12s} | {c.get('seniority_level','?'):6s} | skill={c.get('skill_score',0):.2f}")
        else:
            lines.append("  (none — hire more candidates)")

        if placed:
            lines.append(f"\nALREADY PLACED (status=placed, DO NOT match/let_go these): {[c['id'] for c in placed]}")

        if pipeline:
            lines.append(f"\nINTERVIEWED (status=in_pipeline, call hire_candidate next): {[c['id'] for c in pipeline]}")

        lines.append(f"\nOPEN ROLES needing candidates — {len(open_roles)} roles:")
        for r in open_roles[:6]:
            lines.append(f"  project_id={r['project_id']}  role_id={r['role_id']}  needs={r['type']}  min_skill={r['min_skill']:.2f}")

        if valid_matches:
            lines.append("\nVALID MATCHES — copy these calls exactly:")
            for m in valid_matches[:4]:
                lines.append(
                    f"  confirm_project(project_id=\"{m['project_id']}\")"
                    f"  →  match_candidate_to_project("
                    f"candidate_id=\"{m['candidate_id']}\", "
                    f"project_id=\"{m['project_id']}\", "
                    f"role_id=\"{m['role_id']}\")"
                )
        else:
            needed = {r["type"] for r in open_roles}
            bench_types = {c.get("developer_type") for c in bench}
            lines.append(f"\n(No bench↔role type matches. Roles need: {sorted(needed)}. "
                         f"Bench has: {sorted(bench_types)}. "
                         f"Find and hire a candidate of type in {sorted(needed - bench_types)}.)")

        lines.append("\nTOOL NAME REMINDER: hire_candidate (not 'hired_candidate')")
        action_guide = "\n".join(lines)

        state_str = json.dumps({"candidates": cand_state, "clients": client_state}, indent=2)[:800]
        conversation = [system_msg] + recent + [{
            "role": "user",
            "content": (
                f"[Week {week + 1} of {episode_steps} — current state]\n{state_str}"
                f"{action_guide}\n\n"
                "Continue: hire candidates, fill project roles, then call advance_week."
            ),
>>>>>>> origin/main
        }]

    print(f"  --- Episode done: profit=${final_profit:,.0f}  total_reward={cumulative_env_reward:+,.2f} ---")
    return prompts_out, completions_out, step_rewards, final_profit, cumulative_env_reward


# ---------------------------------------------------------------------------
# OpenEnv-compatible rollout (generic generate_fn interface)
# ---------------------------------------------------------------------------

def rollout_episode(
    env_client,
    model_generate_fn: Callable,
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = 200,
    seed: int | None = None,
) -> tuple[list[dict], float]:
    """
    Generic episode rollout compatible with the OpenEnv CallToolAction API.

    Parameters
    ----------
    env_client        : OpenEnv client (supports .reset() and .step())
    model_generate_fn : callable(prompt_ids) → {"completion_ids": tensor, "logprobs": ...}
    tokenizer         : matching tokenizer
    system_prompt     : system message text
    max_turns         : maximum tool calls per episode
    seed              : optional seed passed to env.reset()

    Returns
    -------
    (rollout_data, cumulative_reward)
    rollout_data is a list of dicts with keys:
        prompt_ids, completion_ids, logprobs, reward, tool_name
    """
    try:
        from openenv.core.env_server.mcp_types import CallToolAction
    except ImportError:
        raise ImportError("openenv-core required for rollout_episode")

    reset_result = env_client.reset(seed=seed)
    obs_text = (
        reset_result.observation.metadata.get("message", "")
        if hasattr(reset_result.observation, "metadata")
        else str(reset_result)
    )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Episode started. Current state:\n{obs_text}\n\nWhat is your first action?"},
    ]

    rollout_data: list[dict] = []
    cumulative_reward = 0.0

    for turn in range(max_turns):
        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")

        outputs = model_generate_fn(prompt_ids)
        completion_ids = outputs["completion_ids"]
        logprobs = outputs.get("logprobs")
        completion_text = _strip_think(
            tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        )

        parsed = parse_tool_call(completion_text)
        reward = 0.0
        done = False

        if parsed:
            tool_name, params = parsed
            try:
                action = CallToolAction(tool_name=tool_name, arguments=params)
                result = env_client.step(action)
                reward = float(result.reward or 0.0)
                cumulative_reward += reward
                obs_meta = (
                    result.observation.metadata
                    if hasattr(result.observation, "metadata")
                    else {}
                )
                tool_output = json.dumps(obs_meta.get("tool_result", obs_meta), indent=2)
                done = result.done
            except Exception as e:
                tool_output = f"Error: {e}"
        else:
            tool_output = (
                "No tool call found. Use the native format:\n"
                '<tool_call>\n{"name": "tool_name", "arguments": {}}\n</tool_call>'
            )

        conversation.append({"role": "assistant", "content": completion_text})
        conversation.append({
            "role": "user",
            "content": (
                f"Tool result:\n{tool_output}\n"
                f"Step reward: {reward:.4f} | Cumulative: {cumulative_reward:.4f}\n"
                + ("Episode DONE." if done else "Continue. What is your next action?")
            ),
        })

        rollout_data.append({
            "prompt_ids": prompt_ids[0].tolist(),
            "completion_ids": completion_ids[0].tolist(),
            "logprobs": logprobs,
            "reward": reward,
            "tool_name": parsed[0] if parsed else "parse_error",
        })

        if done:
            break

    return rollout_data, cumulative_reward
