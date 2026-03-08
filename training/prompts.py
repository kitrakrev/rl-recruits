"""
System prompt, tool schema, and tool-call parser for the Staffing Agency agent.

SYSTEM_PROMPT  — injected as the "system" role in every conversation.
                 Built dynamically from TOOLS so names are always in sync.
TOOLS          — OpenAI-format tool schema list passed to apply_chat_template
parse_tool_call — extracts (tool_name, args) from any Qwen / generic model output
"""
from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# Defined FIRST so SYSTEM_PROMPT can reference the canonical names.
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "get_agency_state",
        "description": "Get overall agency metrics and current week.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_client_state",
        "description": "Get list of clients and their project statuses.",
        "parameters": {"type": "object", "properties": {
            "client_id": {"type": "string", "description": "Optional specific client ID"}}}}},
    {"type": "function", "function": {
        "name": "get_candidate_state",
        "description": "Get lists of hired, interviewing, and available candidates.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_financial_summary",
        "description": "Get detailed cash flow and profit/loss data.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_market_demand",
        "description": "See which roles are currently most requested by clients.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_candidate_types",
        "description": "Returns valid developer_type and seniority_level enum values plus skill/composite score ranges. Call this before find_candidate if unsure of valid filter values.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_available_projects",
        "description": "List all open (non-sealed) projects with their roles, required developer_type, min_skill_score, and bill_rate_weekly. Call this to discover project_id and role_id values before confirming or matching.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_candidate",
        "description": "Search the market for candidates. Call get_candidate_types first to see valid filter values. All filters are optional and combinable.",
        "parameters": {"type": "object", "properties": {
            "developer_type":      {"type": "string",  "description": "e.g. 'backend', 'frontend', 'ml_engineer', 'devops', 'fullstack'"},
            "seniority_level":     {"type": "string",  "description": "e.g. 'junior', 'mid', 'senior'"},
            "min_skill_score":     {"type": "number",  "description": "Minimum skill score (0.0–1.0)"},
            "min_composite_rating":{"type": "number",  "description": "Minimum composite rating (0.0–1.0)"}}}}},
    {"type": "function", "function": {
        "name": "interview_candidate",
        "description": "Perform a technical interview with a candidate to reveal skills and salary. Costs $500. MANDATORY before hire_candidate.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "negotiate_salary",
        "description": "Make a salary offer to an interviewed candidate before hiring.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "offer_weekly": {"type": "number"}},
            "required": ["candidate_id", "offer_weekly"]}}},
    {"type": "function", "function": {
        "name": "hire_candidate",
        "description": "Hire an interviewed candidate to your bench. Costs $2,000 onboarding. interview_candidate MUST be called first or this will fail.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "confirm_project",
        "description": "Commit to a project so you can fill its roles. Call find_available_projects first to get valid project_id values.",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "match_candidate_to_project",
        "description": "Place a hired (status=hired) bench candidate onto a confirmed project role. Types must match. Speed bonus if sealed within 2 weeks.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "project_id":   {"type": "string"},
            "role_id":      {"type": "string"}},
            "required": ["candidate_id", "project_id", "role_id"]}}},
    {"type": "function", "function": {
        "name": "let_go_candidate",
        "description": "Release a benched candidate who is costing more salary than any available bill rate. Costs 2× weekly salary as severance.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "pass_on_project",
        "description": "Decline a project you cannot fill, avoiding an expiry penalty.",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "request_project_extension",
        "description": "Request more time on a project approaching its deadline.",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "advance_week",
        "description": (
            "Finish all actions for the current week and advance simulation time. "
            "This triggers billing, bench costs, project arrivals, and deadline checks."
        ),
        "parameters": {"type": "object", "properties": {}}}},
]

# Canonical set of valid tool names — derived from TOOLS so it's always in sync.
KNOWN_TOOLS: frozenset[str] = frozenset(t["function"]["name"] for t in TOOLS)


# ---------------------------------------------------------------------------
# System prompt — built dynamically so tool names are always correct
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    # Auto-generate the tool reference block from TOOLS so names never drift
    tool_lines = []
    for t in TOOLS:
        fn = t["function"]
        name = fn["name"]
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        if props:
            args = ", ".join(
                f"{k}: {v.get('type','str')}"
                + ("" if k in required else "?")
                for k, v in props.items()
            )
            sig = f"{name}({args})"
        else:
            sig = f"{name}()"
        tool_lines.append(f"  {sig:<60s} — {fn['description'][:80]}")

    tool_block = "\n".join(tool_lines)

    return f"""You are a staffing agency CEO. Maximize profit over 52 weeks by hiring candidates and placing them on client projects.

══════════════════════════════════════════
EXACT WORKFLOW EACH WEEK (follow in order)
══════════════════════════════════════════
STEP 1 — Discover what projects need:
  find_available_projects()
  → returns project_id, role_id, developer_type, min_skill_score, bill_rate_weekly
  → call this FIRST every week so you know what candidate types to hire

STEP 2 — Source matching candidates (types MUST match open roles):
  find_candidate(developer_type="backend")            ← filter by type needed
  interview_candidate(candidate_id="C3")              ← MANDATORY. Costs $500. Reveals skill.
  negotiate_salary(candidate_id="C3", offer_weekly=1200)  ← optional, reduces salary cost
  hire_candidate(candidate_id="C3")                   ← only after interview. Costs $2,000.

  ⚠ hire_candidate WILL FAIL if interview_candidate was not called first.

STEP 3 — Place the candidate (status must be "hired", not "placed"):
  confirm_project(project_id="P2")                    ← commit to project first
  match_candidate_to_project(candidate_id="C3", project_id="P2", role_id="R2-0")

  ⚠ Only candidates with status=hired (on bench) can be matched.
  ⚠ Candidates with status=placed are ALREADY on a project — do NOT try to match them again.

STEP 4 — End the week:
  advance_week()                                      ← billing fires, bench costs fire

══════════════════════════════════════════
CANDIDATE STATUSES — know the difference
══════════════════════════════════════════
  available   → in market, not yours. Call find_candidate → interview → hire.
  in_pipeline → interviewed, not yet hired. Call hire_candidate next.
  hired       → on your bench, READY to match. Call match_candidate_to_project.
  placed      → already on a project, earning revenue. Do NOT match or let_go.

The ACTION GUIDE each week shows your bench (hired) and placed separately.

══════════════════════════════════════════
MATCHING RULE
══════════════════════════════════════════
match_candidate_to_project FAILS if:
  ✗ candidate.developer_type ≠ role.developer_type
  ✗ candidate is not in "hired" status (placed/in_pipeline candidates cannot be matched)
  ✗ project_id or role_id does not exist or is not confirmed

To match successfully:
  ✓ Use EXACT IDs from the BENCH list in the ACTION GUIDE
  ✓ Match candidate type to role type (shown in ACTION GUIDE VALID MATCHES)
  ✓ confirm_project first if not already confirmed

══════════════════════════════════════════
ECONOMICS
══════════════════════════════════════════
Revenue/wk  = bill_rate_weekly per placed candidate
Cost/wk     = salary_weekly per hired candidate (bench or placed)
- Interview:  −$500 one-time
- Hire:       −$2,000 one-time
- Bench burn: −salary_weekly while hired but not placed
- Expired project: large penalty
- Churn (client satisfaction < 0.3): −$50,000

KEY: backend candidate salary=$1,200/wk on role bill=$3,000/wk → +$1,800/wk profit.
     Unplaced candidate salary=$1,200/wk → −$1,200/wk loss.

══════════════════════════════════════════
RULES
══════════════════════════════════════════
- Max 10 actions per week. Call advance_week when done.
- Use ONLY IDs from tool responses or the ACTION GUIDE — never invent IDs.
- negotiate_salary BEFORE hire to maximize margin.
- let_go_candidate for bench candidates you cannot place soon.

══════════════════════════════════════════
ALL AVAILABLE TOOLS (exact names — do not misspell)
══════════════════════════════════════════
{tool_block}
"""


SYSTEM_PROMPT: str = _build_system_prompt()


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------

def _fuzzy_tool_name(name: str) -> str:
    """Return the best matching known tool name for a possibly-misspelled name.

    Uses edit-distance (difflib) so 'hired_candidate' → 'hire_candidate',
    'find_available_candidate' → 'find_candidate', etc.
    Returns the input unchanged if it's already a known tool name or no close
    match exists (similarity < 0.7).
    """
    if name in KNOWN_TOOLS:
        return name
    import difflib
    matches = difflib.get_close_matches(name, KNOWN_TOOLS, n=1, cutoff=0.7)
    return matches[0] if matches else name


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract a (tool_name, arguments) pair from model output.

    Handles three common formats:
      1. Native <tool_call>...</tool_call> tags (Qwen2.5 / Qwen3 standard)
      2. XML-style <function=name><parameter=k>v</parameter></function>
      3. Bare JSON {"name": "...", "arguments": {...}}

    Always picks the LAST match so multi-turn conversation history doesn't
    accidentally re-trigger an earlier tool call.

    Applies fuzzy name correction via _fuzzy_tool_name so minor typos like
    'hired_candidate' are silently fixed to 'hire_candidate'.

    Returns None if no valid tool call is found.
    """
    # Strip <think>...</think> reasoning blocks before parsing
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip Qwen special tokens that sometimes bleed into tool_call blocks
    text = re.sub(r"<\|im_start\|>[^\n]*\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)

    # 1. Native <tool_call>...</tool_call> tags
    all_matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    native_match = all_matches[-1] if all_matches else None
    if native_match:
        # Strip any leftover Qwen role tokens from inside the block
        content = re.sub(r"<\|[^|]+\|>", "", native_match.group(1)).strip()

        # a) JSON format: {"name": "...", "arguments": {...}}
        try:
            data = json.loads(content)
            if "name" in data:
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return (_fuzzy_tool_name(data["name"]), args)
        except Exception:
            pass

        # b) Qwen-Coder XML format: <function=name><parameter=k>v</parameter></function>
        fn_match = re.search(r"<function=(\w+)>(.*?)</function>", content, re.DOTALL)
        if fn_match:
            name = _fuzzy_tool_name(fn_match.group(1))
            params: dict = {}
            for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", fn_match.group(2), re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            return (name, params)

        # c) Bare function name: <function=name /> or <function=name>
        fn_bare = re.search(r"<function=(\w+)\s*/?>", content)
        if fn_bare:
            return (_fuzzy_tool_name(fn_bare.group(1)), {})

    # 2. Bare JSON with name + arguments (no wrapper tags)
    try:
        json_match = re.search(r'\{[^{}]*"name"\s*:[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "name" in data:
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return (_fuzzy_tool_name(data["name"]), args)
    except Exception:
        pass

    return None
