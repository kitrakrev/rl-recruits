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

SYSTEM_PROMPT = """You are a staffing agency CEO managing a recruiting business.
Your goal is to maximise profit over 52 business weeks by:
1. Finding and interviewing candidates from the market
2. Hiring top candidates and placing them on client projects
3. Managing client relationships and filling projects before deadlines
4. Balancing bench costs (hired-but-unplaced = salary drain) vs revenue

MULTI-TURN BUSINESS WEEKS:
- You can take MULTIPLE actions in a single week (interviewing, hiring, matching).
- Time ONLY advances when you call `advance_week`.
- You SHOULD take all necessary actions for the current week and THEN call `advance_week`.
- Do not call `advance_week` until you have finished your business for that week.
- You are limited to 10 actions per week. If you exceed this, the week will advance automatically.

CRITICAL ECONOMICS (updated):
- Interview costs $500 per candidate screened - be selective
- Salary is DYNAMIC: every candidate has a unique salary_expectation (their floor)
  -> Junior backend ~$75k/yr, Senior ML Engineer ~$150k × 1.3 × skill modifier
- Bill rate is VARIABLE per role: what the client pays (set by project, $130k–$300k+/yr)
- TRUE MARGIN = bill_rate_weekly − salary_weekly per placed candidate per week
  -> A cheap junior (salary $1,200/wk) on a $3,000/wk bill-rate role = +$1,800/wk profit
  -> An expensive senior ($3,000/wk) on a $2,800/wk role = −$200/wk LOSS every week
- Benched candidate = −salary_weekly BURN per week (dynamic, not fixed)
- Onboarding: −$2,000 one-time per hire
- Projects must be FULLY filled (SEALED) to lock in recurring revenue
- Expired projects = large penalty; client churn (satisfaction < 0.3) = $50,000 LTV loss

STRATEGY HINTS:
- Use negotiate_salary to lower a candidate's salary before hiring them
- Check bill_rate_weekly on each role before committing a candidate - only place where margin > 0
- Use get_market_demand to identify which developer types are most needed
- Confirm projects before committing candidates (confirm_project)
- Pass on projects where you cannot fill all roles - preventing expiry avoids the penalty
- Let go of benched candidates whose salary exceeds any available bill rate (they lose money)

Use the available tools to manage your agency. Think step by step.
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
        "name": "find_candidate",
        "description": "Search the market for a new candidate by developer type.",
        "parameters": {"type": "object", "properties": {
            "developer_type": {"type": "string", "description": "Optional type e.g. 'Backend', 'Frontend', 'ML Engineer'"}}}}},
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

    # 1. Native <tool_call>...</tool_call> tags
    all_matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    native_match = all_matches[-1] if all_matches else None
    if native_match:
        content = native_match.group(1).strip()

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
