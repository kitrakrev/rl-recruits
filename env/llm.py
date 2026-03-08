"""
LLM layer for the Staffing Agency environment.

Stub mode: returns plausible static/random values — no API key needed.
Live mode: calls Anthropic claude-sonnet-4-6 with structured JSON output.

Toggle via Config.llm_mode = "stub" | "live"
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

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
    client_message: str
    ltv_impact: float


@dataclass
class LeaveDecision:
    leaves: bool
    reason: str
    patience_remaining: int


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _stub_interview(candidate: "Candidate", job_description: str) -> InterviewResult:
    rng = random.Random(hash(candidate.id) ^ hash(job_description[:20]))
    # Base rating correlates with skill_score
    skill = candidate.skill_score
    base = int(_clamp(round(1 + skill * 4 + rng.gauss(0, 0.4)), 1, 5))
    tech = _clamp(skill + rng.gauss(0, 0.1), 0.2, 1.0)
    comm = _clamp(0.6 + rng.gauss(0, 0.15), 0.2, 1.0)
    culture = _clamp(0.65 + rng.gauss(0, 0.12), 0.2, 1.0)
    red_flags = []
    if skill < 0.4:
        red_flags.append("Below-average technical depth")
    if comm < 0.45:
        red_flags.append("Communication concerns noted")
    proceed = base >= 2 and not (skill < 0.3)
    summary = (
        f"{candidate.seniority_level.capitalize()} {candidate.developer_type} candidate. "
        f"Skill={skill:.2f}, base_rating={base}. "
        + ("Recommend advancing." if proceed else "Not recommended.")
    )
    return InterviewResult(
        base_rating=base,
        technical_score=round(tech, 2),
        communication=round(comm, 2),
        culture_fit=round(culture, 2),
        red_flags=red_flags,
        summary=summary,
        proceed=proceed,
    )


def _stub_project_fit(
    candidate: "Candidate", role: "Role", project_context: dict
) -> FitResult:
    rng = random.Random(hash(candidate.id) ^ hash(role.role_id))
    # Fit depends on skill vs min requirement and type match
    skill_gap = candidate.skill_score - role.min_skill_score
    base_fit = _clamp(3 + skill_gap * 4 + rng.gauss(0, 0.3), 1, 5)
    fit_rating = int(round(base_fit))
    fit_rating = max(1, min(5, fit_rating))
    composite = round(0.4 * candidate.base_rating + 0.6 * fit_rating, 2)
    sat_delta = _clamp((fit_rating - 3) * 0.1 + rng.gauss(0, 0.02), -0.3, 0.3)
    risk_flags = []
    if skill_gap < 0.05:
        risk_flags.append("Candidate barely meets minimum skill threshold")
    if candidate.developer_type != role.developer_type:
        risk_flags.append(f"Adjacent type match: {candidate.developer_type} → {role.developer_type}")
    rationale = (
        f"Candidate skill={candidate.skill_score:.2f} vs role min={role.min_skill_score:.2f}. "
        f"Fit rating={fit_rating}. "
        + ("Strong match." if fit_rating >= 4 else "Acceptable match." if fit_rating >= 3 else "Weak match.")
    )
    return FitResult(
        project_fit_rating=fit_rating,
        composite_rating=composite,
        fit_rationale=rationale,
        risk_flags=risk_flags,
        client_satisfaction_delta=round(sat_delta, 3),
    )


def _stub_salary_negotiation(
    candidate: "Candidate", offer: float, market_context: dict
) -> NegotiationResult:
    accepted = offer >= candidate.salary_expectation
    counter = None if accepted else round(candidate.salary_expectation * 1.05, 2)
    reason = (
        "Offer meets or exceeds expectation." if accepted
        else f"Offer ${offer:.0f}/wk below expectation ${candidate.salary_expectation:.0f}/wk."
    )
    patience_impact = 0 if accepted else -1
    return NegotiationResult(
        accepted=accepted,
        counter_offer=counter,
        acceptance_reason=reason,
        patience_impact=patience_impact,
    )


def _stub_client_satisfaction(
    client: "Client", event: dict, history: list
) -> SatisfactionUpdate:
    event_type = event.get("type", "")
    delta_map = {
        "project_sealed":    +0.08,
        "project_expired":   -0.12,
        "adjacent_match":    -0.04,
        "extension_requested": -0.06,
        "project_confirmed": +0.02,
        "fast_fill_bonus":   +0.05,
    }
    base_delta = delta_map.get(event_type, 0.0)
    # History penalty: repeated expiries hurt more
    expiry_count = sum(1 for e in history[-5:] if e.get("type") == "project_expired")
    delta = base_delta - expiry_count * 0.02
    new_score = _clamp(client.satisfaction_score + delta, 0.0, 1.0)
    churn_risk = new_score < 0.3
    messages = {
        "project_sealed": "Great work — the team is in place and we're ready to go.",
        "project_expired": "This delay is unacceptable. We need faster filling.",
        "adjacent_match": "The candidate isn't exactly what we asked for, but we'll see.",
        "extension_requested": "We're disappointed you need more time.",
        "project_confirmed": "Good to hear you're committed to this project.",
        "fast_fill_bonus": "Impressive turnaround — exactly what we needed.",
    }
    client_message = messages.get(event_type, "No comment.")
    ltv_impact = delta * 50_000
    return SatisfactionUpdate(
        new_score=round(new_score, 4),
        delta=round(delta, 4),
        churn_risk=churn_risk,
        client_message=client_message,
        ltv_impact=round(ltv_impact, 2),
    )


def _stub_candidate_leave(
    candidate: "Candidate", agency_context: dict
) -> LeaveDecision:
    leaves = candidate.patience_remaining <= 0
    reason = (
        "Patience exhausted — accepted competing offer." if leaves
        else f"Still waiting. {candidate.patience_remaining} weeks of patience left."
    )
    new_patience = max(0, candidate.patience_remaining - 1)
    return LeaveDecision(
        leaves=leaves,
        reason=reason,
        patience_remaining=new_patience,
    )


# ---------------------------------------------------------------------------
# Live implementations (Anthropic API)
# ---------------------------------------------------------------------------

def _live_call(prompt: str, schema_hint: str, config: "Config") -> dict:
    """Call Anthropic API and parse JSON response."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: uv pip install anthropic")

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    system = (
        "You are a simulation engine for a staffing agency RL environment. "
        "Always respond with valid JSON matching the requested schema. "
        "Be realistic and varied — not all candidates or matches are average."
    )
    message = client.messages.create(
        model=config.llm_model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt + f"\n\nRespond with JSON: {schema_hint}"}],
    )
    text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _live_interview(candidate: "Candidate", job_description: str, config: "Config") -> InterviewResult:
    prompt = f"""
You are a senior technical interviewer at a staffing agency.
Candidate profile:
  - developer_type: {candidate.developer_type}
  - seniority: {candidate.seniority_level}
  - skill_score: {candidate.skill_score:.2f}
Role being considered: {job_description}

Conduct a simulated interview. Be realistic — not all candidates perform well.
"""
    schema = '{"base_rating": int 1-5, "technical_score": float 0-1, "communication": float 0-1, "culture_fit": float 0-1, "red_flags": list[str], "summary": str, "proceed": bool}'
    data = _live_call(prompt, schema, config)
    return InterviewResult(**data)


def _live_project_fit(candidate: "Candidate", role: "Role", project_context: dict, config: "Config") -> FitResult:
    prompt = f"""
Evaluate candidate fit for a project role.
Candidate: developer_type={candidate.developer_type}, seniority={candidate.seniority_level}, skill={candidate.skill_score:.2f}, base_rating={candidate.base_rating}
Role: type={role.developer_type}, seniority={role.seniority}, min_skill={role.min_skill_score:.2f}
Project context: {project_context}

Rate fit 1–5 and explain.
"""
    schema = '{"project_fit_rating": int 1-5, "composite_rating": float, "fit_rationale": str, "risk_flags": list[str], "client_satisfaction_delta": float -1 to 1}'
    data = _live_call(prompt, schema, config)
    return FitResult(**data)


def _live_salary_negotiation(candidate: "Candidate", offer: float, market_context: dict, config: "Config") -> NegotiationResult:
    prompt = f"""
Simulate a candidate's response to a salary offer.
Candidate: {candidate.developer_type}, {candidate.seniority_level}, skill={candidate.skill_score:.2f}
Offer (weekly): ${offer:.2f}
Candidate's minimum expectation (weekly): ${candidate.salary_expectation:.2f}
Market context: {market_context}

Would the candidate accept, reject, or counter?
"""
    schema = '{"accepted": bool, "counter_offer": float or null, "acceptance_reason": str, "patience_impact": int}'
    data = _live_call(prompt, schema, config)
    return NegotiationResult(**data)


def _live_client_satisfaction(client: "Client", event: dict, history: list, config: "Config") -> SatisfactionUpdate:
    prompt = f"""
Simulate client satisfaction update.
Client: industry={client.industry}, current_satisfaction={client.satisfaction_score:.2f}
Recent event: {event}
Last 5 events: {history[-5:]}

Update satisfaction score 0–1 and generate realistic client feedback.
"""
    schema = '{"new_score": float 0-1, "delta": float, "churn_risk": bool, "client_message": str, "ltv_impact": float}'
    data = _live_call(prompt, schema, config)
    return SatisfactionUpdate(**data)


def _live_candidate_leave(candidate: "Candidate", agency_context: dict, config: "Config") -> LeaveDecision:
    prompt = f"""
Simulate whether a benched candidate decides to leave.
Candidate: {candidate.developer_type}, {candidate.seniority_level}, weeks_on_bench={candidate.weeks_on_bench}, patience_remaining={candidate.patience_remaining}
Agency context: {agency_context}

Does the candidate leave?
"""
    schema = '{"leaves": bool, "reason": str, "patience_remaining": int}'
    data = _live_call(prompt, schema, config)
    return LeaveDecision(**data)


# ---------------------------------------------------------------------------
# Public router
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
