"""
System prompt, tool schema, and tool-call parser for the Staffing Agency agent.

SYSTEM_PROMPT  — injected as the "system" role in every conversation
TOOLS          — OpenAI-format tool schema list passed to apply_chat_template
parse_tool_call — extracts (tool_name, args) from any Qwen / generic model output
"""
from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a staffing agency CEO. Maximize profit over 52 weeks by hiring candidates and placing them on client projects.

══════════════════════════════════════════
EXACT WORKFLOW EACH WEEK (follow in order)
══════════════════════════════════════════
STEP 1 — Discover open projects and what types they need:
  find_available_projects()
  → returns list of projects with project_id, role_id, developer_type, min_skill_score, bill_rate_weekly
  → ALWAYS call this first so you know what types of candidates to hire

STEP 2 — Hire candidates that match open role types:
  find_candidate(developer_type="backend")    ← returns IDs of market candidates
  interview_candidate(candidate_id="C3")      ← MANDATORY before hire. Costs $500. Reveals skill.
  negotiate_salary(candidate_id="C3", offer_weekly=1200)  ← optional, lowers salary
  hire_candidate(candidate_id="C3")           ← only works AFTER interview. Costs $2,000.

  ⚠ hire_candidate WILL FAIL with "not found" if you skip interview_candidate first.

STEP 3 — Place the candidate (types MUST match):
  confirm_project(project_id="P2")            ← commit to the project first
  match_candidate_to_project(candidate_id="C3", project_id="P2", role_id="R2-0")
  ← candidate type must equal role developer_type exactly

STEP 4 — End the week:
  advance_week()                              ← billing fires, bench costs fire

══════════════════════════════════════════
CRITICAL MATCHING RULE — READ THIS CAREFULLY
══════════════════════════════════════════
match_candidate_to_project WILL FAIL if:
  ✗ candidate.developer_type ≠ role.developer_type  (e.g. "backend" candidate → "frontend" role)
  ✗ candidate.skill_score < role.min_skill_score
  ✗ candidate is not hired yet
  ✗ project_id or role_id is wrong / not confirmed

To match successfully:
  ✓ Find the candidate's developer_type (shown in find_candidate results)
  ✓ Find a role with the SAME developer_type (shown in the ACTION GUIDE each week)
  ✓ Use the EXACT IDs from the ACTION GUIDE — never invent IDs
  ✓ Confirm the project first with confirm_project(project_id=...)

══════════════════════════════════════════
ECONOMICS
══════════════════════════════════════════
REVENUE per week   = bill_rate_weekly per placed candidate (from project role)
COST per week      = salary_weekly per hired candidate (placed or benched)
PROFIT per week    = revenue − costs
- Interview:  −$500 one-time
- Hire:       −$2,000 one-time onboarding
- Bench burn: −salary_weekly if hired but NOT placed on a project
- Margin:     bill_rate_weekly − salary_weekly (must be POSITIVE to make money)
- Expired project:  large penalty
- Client churn (satisfaction < 0.3): −$50,000

KEY: A backend candidate with salary $1,200/wk on a backend role billing $3,000/wk = +$1,800/wk profit.
     An unplaced candidate at $1,200/wk = −$1,200/wk loss every week.

══════════════════════════════════════════
RULES
══════════════════════════════════════════
- Max 10 actions per week. Call advance_week when done.
- Each week you receive an ACTION GUIDE with real IDs and VALID MATCHES — use them.
- DO NOT invent IDs. Use only IDs from tool responses or the ACTION GUIDE.
- negotiate_salary BEFORE hire to maximize margin.
- let_go_candidate for benched candidates you cannot place.
"""


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
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
        "name": "get_candidate_types",
        "description": "Returns valid developer_type and seniority_level enum values plus skill/composite score ranges. Call this before find_candidate if unsure of valid filter values.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_candidate",
        "description": "Search the market for candidates. Call get_candidate_types first to see valid filter values. All filters are optional and combinable.",
        "parameters": {"type": "object", "properties": {
            "developer_type":      {"type": "string",  "description": "e.g. 'backend', 'frontend', 'ml engineer'"},
            "seniority_level":     {"type": "string",  "description": "e.g. 'junior', 'mid', 'senior'"},
            "min_skill_score":     {"type": "number",  "description": "Minimum skill score (0.0–1.0); requires a prior interview"},
            "min_composite_rating":{"type": "number",  "description": "Minimum composite rating (0.0–1.0); requires a prior interview"}}}}},
    {"type": "function", "function": {
        "name": "interview_candidate",
        "description": "Perform a technical interview with a candidate to reveal skills and salary. Costs $500.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "hire_candidate",
        "description": "Hire an interviewed candidate to your bench. Costs $2,000 onboarding.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "negotiate_salary",
        "description": "Make a salary offer to an interviewed candidate.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "offer_weekly": {"type": "number"}},
            "required": ["candidate_id", "offer_weekly"]}}},
    {"type": "function", "function": {
        "name": "match_candidate_to_project",
        "description": "Place a hired candidate onto a specific project role.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "project_id": {"type": "string"},
            "role_id": {"type": "string"}},
            "required": ["candidate_id", "project_id", "role_id"]}}},
    {"type": "function", "function": {
        "name": "get_financial_summary",
        "description": "Get detailed cash flow and profit/loss data.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_market_demand",
        "description": "See which roles are currently most requested by clients.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_available_projects",
        "description": "List all open (non-sealed) projects with their roles, required developer_type, min_skill_score, and bill_rate_weekly. Call this to discover project_id and role_id values before confirming or matching.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "confirm_project",
        "description": "Commit to a project so you can fill its roles. Call find_available_projects first to get valid project_id values.",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "pass_on_project",
        "description": "Decline a project you cannot fill, avoiding an expiry penalty.",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "let_go_candidate",
        "description": "Release a benched candidate who is costing more salary than any available bill rate. Costs 2× weekly salary as severance.",
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"}}, "required": ["candidate_id"]}}},
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


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract a (tool_name, arguments) pair from model output.

    Handles three common formats:
      1. Native <tool_call>...</tool_call> tags (Qwen2.5 / Qwen3 standard)
      2. XML-style <function=name><parameter=k>v</parameter></function>
      3. Bare JSON {"name": "...", "arguments": {...}}

    Always picks the LAST match so multi-turn conversation history doesn't
    accidentally re-trigger an earlier tool call.

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
                return (data["name"], args)
        except Exception:
            pass

        # b) Qwen-Coder XML format: <function=name><parameter=k>v</parameter></function>
        fn_match = re.search(r"<function=(\w+)>(.*?)</function>", content, re.DOTALL)
        if fn_match:
            name = fn_match.group(1)
            params: dict = {}
            for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", fn_match.group(2), re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            return (name, params)

        # c) Bare function name: <function=name /> or <function=name>
        fn_bare = re.search(r"<function=(\w+)\s*/?>", content)
        if fn_bare:
            return (fn_bare.group(1), {})

    # 2. Bare JSON with name + arguments (no wrapper tags)
    try:
        json_match = re.search(r'\{[^{}]*"name"\s*:[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "name" in data:
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return (data["name"], args)
    except Exception:
        pass

    return None
