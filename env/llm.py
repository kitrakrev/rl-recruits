"""
LLM layer for the Staffing Agency environment.

Stub mode: returns plausible static/random values — no API key needed.
Live mode: calls local open-source model via vLLM (OpenAI compatible API).

Toggle via Config.llm_mode = "stub" | "live"
"""
from __future__ import annotations
import json
import random
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None

if TYPE_CHECKING:
    from .models import Candidate, Role, Client
    from .config import Config


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InterviewResult:
    base_rating: int          # 1–5
    technical_score: float    # 0.0–1.0
    communication: float      # 0.0–1.0
    culture_fit: float        # 0.0–1.0
    red_flags: list
    summary: str
    proceed: bool


@dataclass
class FitResult:
    project_fit_rating: int   # 1–5
    composite_rating: float   # 0.4×base + 0.6×fit
    fit_rationale: str
    risk_flags: list
    client_satisfaction_delta: float   # −1.0 to +1.0


@dataclass
class NegotiationResult:
    accepted: bool
    counter_offer: Optional[float]
    acceptance_reason: str
    patience_impact: int      # Δ patience_remaining


@dataclass
class SatisfactionUpdate:
    new_score: float
    delta: float
    churn_risk: bool


@dataclass
class LeaveDecision:
    leave: bool
    reason: str


# ---------------------------------------------------------------------------
# Live LLM Calls (via Local vLLM / OpenAI API)
# ---------------------------------------------------------------------------

def _get_client():
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: uv pip install openai")
    # Points to local vLLM server by default (port 8001)
    base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8001/v1")
    return OpenAI(base_url=base_url, api_key="sk-local-dev")

def _get_async_client():
    if AsyncOpenAI is None:
        raise ImportError("openai package not installed. Run: uv pip install openai")
    base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8001/v1")
    return AsyncOpenAI(base_url=base_url, api_key="sk-local-dev")

def _get_model_name():
    # Use whatever model is loaded in vLLM, or override with env var
    return os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")

def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Helper to call local LLM and return parsed JSON."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=_get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0, # Zero temp for deterministic judging
            max_tokens=500
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\\n[!] LLM Call Failed: {e}")
        print("[!] Returning empty dict fallback to prevent crash.")
        return {}

async def _async_call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    client = _get_async_client()
    try:
        response = await client.chat.completions.create(
            model=_get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\n[!] Async LLM Call Failed: {e}")
        print("[!] Returning empty dict fallback to prevent crash.")
        return {}

def _live_interview(candidate: "Candidate", job_desc: str, config: "Config") -> InterviewResult:
    sys_prompt = """You are an expert technical interviewer evaluating a candidate for a staffing agency.
    Output your evaluation strictly as a JSON object matching this structure:
    {
      "base_rating": int (1 to 5),
      "technical_score": float (0.0 to 1.0),
      "communication": float (0.0 to 1.0),
      "culture_fit": float (0.0 to 1.0),
      "red_flags": [list of strings, max 2],
      "summary": "string summary",
      "proceed": boolean (true if rating >= 3)
    }"""
    
    user_prompt = f"Candidate is a {candidate.seniority_level} {candidate.developer_type} with a hidden skill score of {candidate.skill_score}. Evaluate them for a general {job_desc} role."
    
    res = _call_llm_json(sys_prompt, user_prompt)
    
    return InterviewResult(
        base_rating=res.get("base_rating", 3),
        technical_score=res.get("technical_score", 0.5),
        communication=res.get("communication", 0.5),
        culture_fit=res.get("culture_fit", 0.5),
        red_flags=res.get("red_flags", []),
        summary=res.get("summary", "Fallback summary due to parsing error."),
        proceed=res.get("proceed", True)
    )


async def _asynclive_interview(candidate: "Candidate", job_desc: str, config: "Config") -> InterviewResult:
    sys_prompt = """You are an expert technical interviewer evaluating a candidate for a staffing agency.
    Output your evaluation strictly as a JSON object matching this structure:
    {
      "base_rating": int (1 to 5),
      "technical_score": float (0.0 to 1.0),
      "communication": float (0.0 to 1.0),
      "culture_fit": float (0.0 to 1.0),
      "red_flags": [list of strings, max 2],
      "summary": "string summary",
      "proceed": boolean (true if rating >= 3)
    }"""
    
    user_prompt = f"Candidate is a {candidate.seniority_level} {candidate.developer_type} with a hidden skill score of {candidate.skill_score}. Evaluate them for a general {job_desc} role."
    
    res = await _async_call_llm_json(sys_prompt, user_prompt)
    
    return InterviewResult(
        base_rating=res.get("base_rating", 3),
        technical_score=res.get("technical_score", 0.5),
        communication=res.get("communication", 0.5),
        culture_fit=res.get("culture_fit", 0.5),
        red_flags=res.get("red_flags", []),
        summary=res.get("summary", "Fallback summary due to parsing error."),
        proceed=res.get("proceed", True)
    )

def _live_project_fit(candidate: "Candidate", role: "Role", project_context: dict, config: "Config") -> FitResult:
    sys_prompt = """You are evaluating how well a candidate fits a specific client project.
    Output strictly as a JSON object:
    {
      "project_fit_rating": int (1 to 5),
      "fit_rationale": "string rationale",
      "risk_flags": [list of strings, empty if none],
      "client_satisfaction_delta": float (-0.2 to 0.2)
    }"""
    
    user_prompt = f"""
    Candidate: {candidate.seniority_level} {candidate.developer_type}, Base Rating: {candidate.base_rating}
    Role needed: {role.seniority} {role.developer_type}
    Client Industry: {project_context.get('client_industry', 'Unknown')}
    Evaluate the specific fit and risks."""

    res = _call_llm_json(sys_prompt, user_prompt)
    
    fit_rating = res.get("project_fit_rating", 3)
    comp_rating = round(0.4 * candidate.base_rating + 0.6 * fit_rating, 2)
    
    return FitResult(
        project_fit_rating=fit_rating,
        composite_rating=comp_rating,
        fit_rationale=res.get("fit_rationale", "Standard fit."),
        risk_flags=res.get("risk_flags", []),
        client_satisfaction_delta=res.get("client_satisfaction_delta", 0.0)
    )


async def _asynclive_project_fit(candidate: "Candidate", role: "Role", project_context: dict, config: "Config") -> FitResult:
    sys_prompt = """You are evaluating how well a candidate fits a specific client project.
    Output strictly as a JSON object:
    {
      "project_fit_rating": int (1 to 5),
      "fit_rationale": "string rationale",
      "risk_flags": [list of strings, empty if none],
      "client_satisfaction_delta": float (-0.2 to 0.2)
    }"""
    
    user_prompt = f"""
    Candidate: {candidate.seniority_level} {candidate.developer_type}, Base Rating: {candidate.base_rating}
    Role needed: {role.seniority} {role.developer_type}
    Client Industry: {project_context.get('client_industry', 'Unknown')}
    Evaluate the specific fit and risks."""

    res = await _async_call_llm_json(sys_prompt, user_prompt)
    
    fit_rating = res.get("project_fit_rating", 3)
    comp_rating = round(0.4 * candidate.base_rating + 0.6 * fit_rating, 2)
    
    return FitResult(
        project_fit_rating=fit_rating,
        composite_rating=comp_rating,
        fit_rationale=res.get("fit_rationale", "Standard fit."),
        risk_flags=res.get("risk_flags", []),
        client_satisfaction_delta=res.get("client_satisfaction_delta", 0.0)
    )

def _live_salary_negotiation(candidate: "Candidate", offer: float, market_context: dict, config: "Config") -> NegotiationResult:
    sys_prompt = """You are simulating a candidate responding to a salary offer.
    Output strictly as a JSON object:
    {
      "accepted": boolean,
      "counter_offer": float or null,
      "acceptance_reason": "string",
      "patience_impact": int (-2 to 0)
    }"""
    
    user_prompt = f"""
    Candidate expects ~${candidate.salary_expectation:.2f}/wk.
    Agency is offering ${offer:.2f}/wk.
    Candidate patience left: {candidate.patience_remaining} weeks.
    Decide if they accept, reject, or counter."""

    res = _call_llm_json(sys_prompt, user_prompt)
    
    return NegotiationResult(
        accepted=res.get("accepted", offer >= candidate.salary_expectation),
        counter_offer=res.get("counter_offer"),
        acceptance_reason=res.get("acceptance_reason", "Offer was evaluated."),
        patience_impact=res.get("patience_impact", -1)
    )


async def _asynclive_salary_negotiation(candidate: "Candidate", offer: float, market_context: dict, config: "Config") -> NegotiationResult:
    sys_prompt = """You are simulating a candidate responding to a salary offer.
    Output strictly as a JSON object:
    {
      "accepted": boolean,
      "counter_offer": float or null,
      "acceptance_reason": "string",
      "patience_impact": int (-2 to 0)
    }"""
    
    user_prompt = f"""
    Candidate expects ~${candidate.salary_expectation:.2f}/wk.
    Agency is offering ${offer:.2f}/wk.
    Candidate patience left: {candidate.patience_remaining} weeks.
    Decide if they accept, reject, or counter."""

    res = await _async_call_llm_json(sys_prompt, user_prompt)
    
    return NegotiationResult(
        accepted=res.get("accepted", offer >= candidate.salary_expectation),
        counter_offer=res.get("counter_offer"),
        acceptance_reason=res.get("acceptance_reason", "Offer was evaluated."),
        patience_impact=res.get("patience_impact", -1)
    )

def _live_client_satisfaction(client: "Client", event: dict, history: list, config: "Config") -> SatisfactionUpdate:
    sys_prompt = """You are simulating a corporate client's satisfaction score.
    Output strictly as a JSON object:
    {
      "new_score": float (0.0 to 1.0),
      "delta": float,
      "churn_risk": boolean (true if new_score < 0.3)
    }"""
    
    user_prompt = f"""
    Client Current Satisfaction: {client.satisfaction_score}
    New Event: {event}
    Determine the new satisfaction score based on this event and their history."""

    res = _call_llm_json(sys_prompt, user_prompt)
    new_score = res.get("new_score", client.satisfaction_score)
    
    return SatisfactionUpdate(
        new_score=new_score,
        delta=res.get("delta", 0.0),
        churn_risk=res.get("churn_risk", new_score < config.churn_threshold)
    )


async def _asynclive_client_satisfaction(client: "Client", event: dict, history: list, config: "Config") -> SatisfactionUpdate:
    sys_prompt = """You are simulating a corporate client's satisfaction score.
    Output strictly as a JSON object:
    {
      "new_score": float (0.0 to 1.0),
      "delta": float,
      "churn_risk": boolean (true if new_score < 0.3)
    }"""
    
    user_prompt = f"""
    Client Current Satisfaction: {client.satisfaction_score}
    New Event: {event}
    Determine the new satisfaction score based on this event and their history."""

    res = await _async_call_llm_json(sys_prompt, user_prompt)
    new_score = res.get("new_score", client.satisfaction_score)
    
    return SatisfactionUpdate(
        new_score=new_score,
        delta=res.get("delta", 0.0),
        churn_risk=res.get("churn_risk", new_score < config.churn_threshold)
    )

def _live_candidate_leave(candidate: "Candidate", agency_context: dict, config: "Config") -> LeaveDecision:
    sys_prompt = """You are deciding if a benched candidate will quit the agency.
    Output strictly as a JSON object:
    {
      "leave": boolean,
      "reason": "string"
    }"""
    
    user_prompt = f"""
    Candidate {candidate.seniority_level} {candidate.developer_type} has been on the bench for {candidate.weeks_on_bench} weeks.
    Patience remaining: {candidate.patience_remaining}.
    Do they quit today?"""

    res = _call_llm_json(sys_prompt, user_prompt)
    
    return LeaveDecision(
        leave=res.get("leave", candidate.patience_remaining <= 0),
        reason=res.get("reason", "Ran out of patience.")
    )



async def _asynclive_candidate_leave(candidate: "Candidate", agency_context: dict, config: "Config") -> LeaveDecision:
    sys_prompt = """You are deciding if a benched candidate will quit the agency.
    Output strictly as a JSON object:
    {
      "leave": boolean,
      "reason": "string"
    }"""
    
    user_prompt = f"""
    Candidate {candidate.seniority_level} {candidate.developer_type} has been on the bench for {candidate.weeks_on_bench} weeks.
    Patience remaining: {candidate.patience_remaining}.
    Do they quit today?"""

    res = await _async_call_llm_json(sys_prompt, user_prompt)
    
    return LeaveDecision(
        leave=res.get("leave", candidate.patience_remaining <= 0),
        reason=res.get("reason", "Ran out of patience.")
    )


# ---------------------------------------------------------------------------
# Stub Implementations (Fallback Logic)
# ---------------------------------------------------------------------------

def _stub_interview(candidate: "Candidate", job_desc: str) -> InterviewResult:
    base = int(round(candidate.skill_score * 4)) + 1
    return InterviewResult(
        base_rating=base,
        technical_score=candidate.skill_score,
        communication=random.uniform(0.5, 1.0),
        culture_fit=random.uniform(0.4, 1.0),
        red_flags=["Jumped jobs frequently"] if random.random() < 0.1 else [],
        summary="Stub interview completed.",
        proceed=(base >= 3)
    )


async def _asyncstub_interview(candidate: "Candidate", job_desc: str) -> InterviewResult:
    import asyncio
    await asyncio.sleep(0)
    base = int(round(candidate.skill_score * 4)) + 1
    return InterviewResult(
        base_rating=base,
        technical_score=candidate.skill_score,
        communication=random.uniform(0.5, 1.0),
        culture_fit=random.uniform(0.4, 1.0),
        red_flags=["Jumped jobs frequently"] if random.random() < 0.1 else [],
        summary="Stub interview completed.",
        proceed=(base >= 3)
    )

def _stub_project_fit(candidate: "Candidate", role: "Role", project_context: dict) -> FitResult:
    fit = random.randint(3, 5)
    comp = round(0.4 * candidate.base_rating + 0.6 * fit, 2)
    return FitResult(
        project_fit_rating=fit,
        composite_rating=comp,
        fit_rationale="Stub fit rationale.",
        risk_flags=[],
        client_satisfaction_delta=0.05
    )


async def _asyncstub_project_fit(candidate: "Candidate", role: "Role", project_context: dict) -> FitResult:
    import asyncio
    await asyncio.sleep(0)
    fit = random.randint(3, 5)
    comp = round(0.4 * candidate.base_rating + 0.6 * fit, 2)
    return FitResult(
        project_fit_rating=fit,
        composite_rating=comp,
        fit_rationale="Stub fit rationale.",
        risk_flags=[],
        client_satisfaction_delta=0.05
    )

def _stub_salary_negotiation(candidate: "Candidate", offer: float, market_context: dict) -> NegotiationResult:
    if offer >= candidate.salary_expectation * 0.95:
        return NegotiationResult(True, None, "Offer acceptable.", 0)
    return NegotiationResult(False, candidate.salary_expectation, "Offer too low.", -1)


async def _asyncstub_salary_negotiation(candidate: "Candidate", offer: float, market_context: dict) -> NegotiationResult:
    import asyncio
    await asyncio.sleep(0)
    if offer >= candidate.salary_expectation * 0.95:
        return NegotiationResult(True, None, "Offer acceptable.", 0)
    return NegotiationResult(False, candidate.salary_expectation, "Offer too low.", -1)

def _stub_client_satisfaction(client: "Client", event: dict, history: list) -> SatisfactionUpdate:
    delta = 0.05 if event.get("type") in ("project_sealed", "project_confirmed") else -0.1
    new_s = max(0.0, min(1.0, client.satisfaction_score + delta))
    return SatisfactionUpdate(new_s, delta, new_s < 0.3)


async def _asyncstub_client_satisfaction(client: "Client", event: dict, history: list) -> SatisfactionUpdate:
    import asyncio
    await asyncio.sleep(0)
    delta = 0.05 if event.get("type") in ("project_sealed", "project_confirmed") else -0.1
    new_s = max(0.0, min(1.0, client.satisfaction_score + delta))
    return SatisfactionUpdate(new_s, delta, new_s < 0.3)

def _stub_candidate_leave(candidate: "Candidate", agency_context: dict) -> LeaveDecision:
    leave = candidate.patience_remaining <= 0
    return LeaveDecision(leave, "Ran out of patience." if leave else "")



async def _asyncstub_candidate_leave(candidate: "Candidate", agency_context: dict) -> LeaveDecision:
    import asyncio
    await asyncio.sleep(0)
    leave = candidate.patience_remaining <= 0
    return LeaveDecision(leave, "Ran out of patience." if leave else "")


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------

class LLMRouter:
    def __init__(self, config: "Config"):
        self.config = config

    @property
    def mode(self) -> str:
        return self.config.llm_mode

    def interview(self, candidate: "Candidate", job_description: str) -> InterviewResult:
        if self.mode == "live":
            return _live_interview(candidate, job_description, self.config)
        return _stub_interview(candidate, job_description)

    def project_fit(self, candidate: "Candidate", role: "Role", project_context: dict) -> FitResult:
        if self.mode == "live":
            return _live_project_fit(candidate, role, project_context, self.config)
        return _stub_project_fit(candidate, role, project_context)

    def salary_negotiation(self, candidate: "Candidate", offer: float, market_context: dict) -> NegotiationResult:
        if self.mode == "live":
            return _live_salary_negotiation(candidate, offer, market_context, self.config)
        return _stub_salary_negotiation(candidate, offer, market_context)

    def client_satisfaction(self, client: "Client", event: dict, history: list) -> SatisfactionUpdate:
        if self.mode == "live":
            return _live_client_satisfaction(client, event, history, self.config)
        return _stub_client_satisfaction(client, event, history)

    def candidate_leave(self, candidate: "Candidate", agency_context: dict) -> LeaveDecision:
        if self.mode == "live":
            return _live_candidate_leave(candidate, agency_context, self.config)
        return _stub_candidate_leave(candidate, agency_context)

    async def async_interview(self, candidate: "Candidate", job_description: str) -> InterviewResult:
        if self.mode == "live":
            return await _async_live_interview(candidate, job_description, self.config)
        return await _async_stub_interview(candidate, job_description)

    async def async_project_fit(self, candidate: "Candidate", role: "Role", project_context: dict) -> FitResult:
        if self.mode == "live":
            return await _async_live_project_fit(candidate, role, project_context, self.config)
        return await _async_stub_project_fit(candidate, role, project_context)

    async def async_salary_negotiation(self, candidate: "Candidate", offer: float, market_context: dict) -> NegotiationResult:
        if self.mode == "live":
            return await _async_live_salary_negotiation(candidate, offer, market_context, self.config)
        return await _async_stub_salary_negotiation(candidate, offer, market_context)

    async def async_client_satisfaction(self, client: "Client", event: dict, history: list) -> SatisfactionUpdate:
        if self.mode == "live":
            return await _async_live_client_satisfaction(client, event, history, self.config)
        return await _async_stub_client_satisfaction(client, event, history)

    async def async_candidate_leave(self, candidate: "Candidate", agency_context: dict) -> LeaveDecision:
        if self.mode == "live":
            return await _async_live_candidate_leave(candidate, agency_context, self.config)
        return await _async_stub_candidate_leave(candidate, agency_context)
