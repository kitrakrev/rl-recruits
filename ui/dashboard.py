"""
Staffing Agency RL — Live Training Dashboard
============================================
Real-time Gradio UI that:
  1. Shows live simulation state as the agent trains
  2. Plots reward curves, financial P&L, candidate pipeline, client health
  3. Shows agent action history, tool-call frequency, reward breakdown
  4. Lets the user manually call any tool (override state mid-episode)
  5. Exposes config sliders so the user can tune the environment

Run (with env server already running on :8000):
    python ui/dashboard.py

Or launch server + UI together:
    python ui/dashboard.py --start_server
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import gradio as gr

# ── optional imports (graceful degradation) ──────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# Global state (ring buffers for charts)
# ─────────────────────────────────────────────────────────────────────────────
MAX_HISTORY = 200

_history: dict[str, deque] = {
    "steps":            deque(maxlen=MAX_HISTORY),
    "cash":             deque(maxlen=MAX_HISTORY),
    "profit":           deque(maxlen=MAX_HISTORY),
    "reward":           deque(maxlen=MAX_HISTORY),
    "burn_rate":        deque(maxlen=MAX_HISTORY),
    "placed":           deque(maxlen=MAX_HISTORY),
    "hired":            deque(maxlen=MAX_HISTORY),
    "benched":          deque(maxlen=MAX_HISTORY),
    "in_pipeline":      deque(maxlen=MAX_HISTORY),
    "avg_satisfaction": deque(maxlen=MAX_HISTORY),
    "placement_rate":   deque(maxlen=MAX_HISTORY),
    "open_slots":       deque(maxlen=MAX_HISTORY),
}

# Agent action tracking
_agent_actions: deque = deque(maxlen=500)   # (step, tool_name, success, reward)
_tool_counts: dict[str, int] = defaultdict(int)
_tool_rewards: dict[str, list[float]] = defaultdict(list)
_episode_stats: list[dict] = []   # summary per episode
_candidate_lifecycle: deque = deque(maxlen=200)  # events: hired, placed, left

_server_url = "http://localhost:8000"
_poll_interval = 2.0
_last_full_state: dict = {}
_log_lines: deque = deque(maxlen=100)
_current_episode = 0
_episode_start_cash = 50_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Server communication helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    if not HAS_REQUESTS:
        return {"error": "requests not installed"}
    try:
        r = requests.post(f"{_server_url}{endpoint}", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _get(endpoint: str) -> dict:
    if not HAS_REQUESTS:
        return {"error": "requests not installed"}
    try:
        r = requests.get(f"{_server_url}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def call_tool(tool_name: str, params: dict | None = None) -> dict:
    payload = {"tool": tool_name, "params": params or {}}
    return _post("/step", payload)


def reset_env(seed: int | None = None) -> dict:
    payload = {"seed": seed} if seed is not None else {}
    return _post("/reset", payload)


def health_check() -> bool:
    try:
        if not HAS_REQUESTS:
            return False
        r = requests.get(f"{_server_url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Fetch full state
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(r: dict) -> dict:
    """Extract tool_result from HTTP step response."""
    if isinstance(r, dict):
        obs = r.get("observation", {})
        meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
        tr = meta.get("tool_result", {})
        return tr if tr else r
    return {}


def fetch_full_state() -> dict:
    global _last_full_state
    state = {}

    agency  = call_tool("get_agency_state")
    cands   = call_tool("get_candidate_state")
    clients = call_tool("get_client_state")
    demand  = call_tool("get_market_demand")
    finance = call_tool("get_financial_summary")

    state["agency"]  = _unwrap(agency)
    state["cands"]   = _unwrap(cands)
    state["clients"] = _unwrap(clients)
    state["demand"]  = _unwrap(demand)
    state["finance"] = _unwrap(finance)
    state["step"]    = state["agency"].get("step", 0)

    # Append to history
    a = state["agency"]
    if a:
        _history["steps"].append(a.get("step", 0))
        _history["cash"].append(a.get("cash_balance", 0))
        _history["profit"].append(a.get("current_profit", 0))
        _history["reward"].append(a.get("current_profit", 0) / 1000)
        _history["burn_rate"].append(a.get("burn_rate", 0))
        _history["placed"].append(a.get("num_candidates_placed", 0))
        _history["hired"].append(a.get("num_candidates_hired", 0))
        _history["benched"].append(a.get("num_candidates_benched", 0))
        _history["in_pipeline"].append(a.get("num_candidates_in_interview", 0))
        _history["placement_rate"].append(a.get("placement_rate", 0))

    clist = state["clients"].get("clients", [])
    if clist:
        avg_sat = sum(c.get("satisfaction_score", 0) for c in clist) / len(clist)
        _history["avg_satisfaction"].append(avg_sat)

    dm = state["demand"].get("total_open_slots", 0)
    _history["open_slots"].append(dm)

    _last_full_state = state
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#cbd5e1"),
    margin=dict(l=44, r=20, t=44, b=30),
)


def _apply_dark(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(height=height, **_CHART_LAYOUT)
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


def chart_financials() -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Cash Balance ($)", "Net Profit ($)"))
    steps = list(_history["steps"])
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=list(_history["cash"]),
            fill="tozeroy", line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56,189,248,0.12)", name="Cash",
        ), row=1, col=1)
        profit = list(_history["profit"])
        colors = ["#ef4444" if p < 0 else "#22c55e" for p in profit]
        fig.add_trace(go.Bar(
            x=steps, y=profit, marker_color=colors, name="Profit",
        ), row=1, col=2)
    fig.update_layout(showlegend=False)
    return _apply_dark(fig)


def chart_candidates() -> go.Figure:
    fig = go.Figure()
    steps = list(_history["steps"])
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=list(_history["placed"]),
            name="Placed", line=dict(color="#22c55e", width=2), stackgroup="one",
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=list(_history["benched"]),
            name="Benched (bench burn)", line=dict(color="#f59e0b", width=2), stackgroup="one",
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=list(_history["in_pipeline"]),
            name="In pipeline", line=dict(color="#38bdf8", width=2), stackgroup="one",
        ))
    fig.update_layout(
        title="Candidate Pipeline (stacked)", showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.25),
    )
    return _apply_dark(fig)


def chart_satisfaction() -> go.Figure:
    fig = go.Figure()
    steps = list(_history["steps"])
    sat = list(_history["avg_satisfaction"])
    if steps and sat:
        colors = ["#ef4444" if s < 0.3 else "#f59e0b" if s < 0.6 else "#22c55e" for s in sat]
        fig.add_trace(go.Scatter(
            x=steps[:len(sat)], y=sat, mode="lines+markers",
            line=dict(color="#a78bfa", width=2),
            marker=dict(color=colors, size=6),
            name="Avg Satisfaction",
        ))
        fig.add_hline(y=0.3, line_dash="dash", line_color="#ef4444",
                      annotation_text="Churn risk", annotation_font_color="#ef4444")
    fig.update_layout(title="Client Satisfaction", yaxis_range=[0, 1])
    return _apply_dark(fig)


def chart_burn_runway() -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Burn Rate ($/wk)", "Cash Runway (wks)"))
    steps = list(_history["steps"])
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=list(_history["burn_rate"]),
            fill="tozeroy", line=dict(color="#f87171", width=2),
            fillcolor="rgba(248,113,113,0.12)", name="Burn",
        ), row=1, col=1)
        runway = [
            (c / b) if b > 0 else 52
            for c, b in zip(_history["cash"], _history["burn_rate"])
        ]
        fig.add_trace(go.Scatter(
            x=steps, y=runway,
            line=dict(color="#fbbf24", width=2), name="Runway",
        ), row=1, col=2)
        fig.add_hrect(y0=0, y1=4, fillcolor="rgba(239,68,68,0.08)",
                      row=1, col=2, annotation_text="Danger zone")
    fig.update_layout(showlegend=False)
    return _apply_dark(fig)


def chart_market_demand(demand_dict: dict) -> go.Figure:
    if not demand_dict:
        fig = go.Figure()
        fig.update_layout(title="Market Demand")
        return _apply_dark(fig)
    types = list(demand_dict.keys())
    vals  = [demand_dict[t] for t in types]
    COLORS = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#e879f9"]
    fig = go.Figure(go.Bar(
        x=types, y=vals, marker_color=COLORS[:len(types)],
        text=vals, textposition="auto",
    ))
    fig.update_layout(title="Open Role Slots by Developer Type")
    return _apply_dark(fig)


def chart_reward_history() -> go.Figure:
    """Cumulative reward and per-step reward over the episode."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Per-Step Reward", "Cumulative Reward"))
    steps = list(_history["steps"])
    if steps and _agent_actions:
        # Per-step reward from agent actions
        action_steps = [a[0] for a in _agent_actions]
        action_rewards = [a[3] for a in _agent_actions]
        fig.add_trace(go.Bar(
            x=action_steps, y=action_rewards,
            marker_color=["#22c55e" if r >= 0 else "#ef4444" for r in action_rewards],
            name="Step Reward",
        ), row=1, col=1)
        # Cumulative
        cum = []
        total = 0.0
        for r in action_rewards:
            total += r
            cum.append(total)
        fig.add_trace(go.Scatter(
            x=action_steps, y=cum,
            line=dict(color="#38bdf8", width=2), fill="tozeroy",
            fillcolor="rgba(56,189,248,0.10)", name="Cumulative",
        ), row=1, col=2)
    fig.update_layout(showlegend=False)
    return _apply_dark(fig)


def chart_tool_usage() -> go.Figure:
    """Bar chart of tool call frequency."""
    if not _tool_counts:
        fig = go.Figure()
        fig.update_layout(title="Tool Call Frequency (no data yet)")
        return _apply_dark(fig, height=320)
    tools = list(_tool_counts.keys())
    counts = [_tool_counts[t] for t in tools]
    COLORS = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#e879f9",
              "#f87171", "#fbbf24", "#a78bfa", "#22c55e", "#94a3b8"]
    fig = go.Figure(go.Bar(
        y=tools, x=counts,
        orientation="h",
        marker_color=COLORS[:len(tools)],
        text=counts, textposition="auto",
    ))
    fig.update_layout(title="Tool Call Frequency", yaxis=dict(autorange="reversed"))
    return _apply_dark(fig, height=max(300, len(tools) * 28))


def chart_tool_avg_reward() -> go.Figure:
    """Average reward per tool call."""
    if not _tool_rewards:
        fig = go.Figure()
        fig.update_layout(title="Avg Reward per Tool")
        return _apply_dark(fig)
    tools = [t for t in _tool_rewards if _tool_rewards[t]]
    avgs  = [sum(_tool_rewards[t]) / len(_tool_rewards[t]) for t in tools]
    colors = ["#22c55e" if a >= 0 else "#ef4444" for a in avgs]
    fig = go.Figure(go.Bar(
        y=tools, x=avgs, orientation="h",
        marker_color=colors, text=[f"{a:.1f}" for a in avgs], textposition="auto",
    ))
    fig.update_layout(title="Avg Reward per Tool Call", yaxis=dict(autorange="reversed"))
    return _apply_dark(fig, height=max(300, len(tools) * 28))


def chart_placement_rate() -> go.Figure:
    """Placement rate and open demand over time."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Placement Rate (%)", "Open Market Slots"))
    steps = list(_history["steps"])
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=[r * 100 for r in _history["placement_rate"]],
            line=dict(color="#22c55e", width=2),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.10)",
            name="Placement Rate",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=steps[:len(_history["open_slots"])],
            y=list(_history["open_slots"]),
            line=dict(color="#f59e0b", width=2),
            name="Open Slots",
        ), row=1, col=2)
    fig.update_layout(showlegend=False)
    return _apply_dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, color: str = "#38bdf8", sub: str = "") -> str:
    return f"""
<div style="background:#1e293b;border-radius:10px;padding:14px 18px;min-width:130px;flex:1">
  <div style="color:#94a3b8;font-size:11px;margin-bottom:4px;letter-spacing:0.5px">{label}</div>
  <div style="color:{color};font-size:22px;font-weight:700">{value}</div>
  {f'<div style="color:#64748b;font-size:10px;margin-top:3px">{sub}</div>' if sub else ''}
</div>"""


def build_kpi_row(state: dict) -> str:
    a = state.get("agency", {})
    f = state.get("finance", {})
    step    = a.get("step", 0)
    total   = a.get("episode_steps", 52)
    cash    = a.get("cash_balance", 0)
    profit  = f.get("profit", 0)
    placed  = a.get("num_candidates_placed", 0)
    benched = a.get("num_candidates_benched", 0)
    runway  = a.get("cash_runway_weeks", 0)
    prate   = a.get("placement_rate", 0)
    burn    = a.get("burn_rate", 0)
    cum_r   = sum(a[3] for a in _agent_actions) if _agent_actions else 0.0

    profit_color = "#22c55e" if profit >= 0 else "#ef4444"
    runway_color = "#ef4444" if runway < 4 else "#f59e0b" if runway < 8 else "#22c55e"
    reward_color = "#22c55e" if cum_r >= 0 else "#ef4444"

    return f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;padding:4px 0">
  {_kpi("Episode Step",  f"{step} / {total}", "#a78bfa", f"{int(step/total*100)}% complete" if total else "")}
  {_kpi("Cash Balance",  f"${cash:,.0f}", "#38bdf8")}
  {_kpi("Net Profit",    f"${profit:,.0f}", profit_color)}
  {_kpi("Cumul Reward",  f"{cum_r:,.1f}", reward_color)}
  {_kpi("Placed",        str(placed), "#22c55e", "generating revenue")}
  {_kpi("Benched",       str(benched), "#f59e0b", f"${burn:,.0f}/wk burn")}
  {_kpi("Placement %",   f"{prate:.0%}", "#818cf8")}
  {_kpi("Runway",        f"{runway:.1f} wks", runway_color)}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Table / card builders
# ─────────────────────────────────────────────────────────────────────────────

def build_candidates_table(state: dict) -> str:
    cands = state.get("cands", {}).get("candidates", [])
    if not cands:
        return "<p style='color:#64748b;padding:12px'>No candidates on payroll.</p>"
    rows = []
    for c in cands[:30]:
        status = c.get("status", "?")
        color = {"placed": "#22c55e", "hired": "#f59e0b", "in_pipeline": "#38bdf8",
                 "pending_hire": "#a78bfa"}.get(status, "#94a3b8")
        patience = c.get("patience_remaining", "?")
        pat_color = "#ef4444" if isinstance(patience, int) and patience <= 2 else "#94a3b8"
        proj = c.get("assigned_project") or "—"
        ttp  = c.get("contract_weeks_left")
        ttp_str = f"{ttp}w left" if ttp is not None else "—"
        rows.append(f"""
<tr style="border-bottom:1px solid #1e293b">
  <td style="padding:5px 8px;color:#e2e8f0;font-family:monospace;font-size:11px">{c.get('id','?')[:16]}</td>
  <td style="padding:5px 8px;color:#94a3b8;font-size:12px">{c.get('developer_type','?')}</td>
  <td style="padding:5px 8px;color:#94a3b8;font-size:12px">{c.get('seniority_level','?')}</td>
  <td style="padding:5px 8px;color:#a78bfa;font-size:12px">{c.get('skill_score',0):.2f}</td>
  <td style="padding:5px 8px"><span style="color:{color};background:{color}22;padding:2px 7px;border-radius:4px;font-size:10px">{status}</span></td>
  <td style="padding:5px 8px;color:#38bdf8;font-size:12px">{c.get('composite_rating',0):.1f} ★</td>
  <td style="padding:5px 8px;color:#22c55e;font-size:12px">${c.get('margin_weekly',0):,.0f}/wk</td>
  <td style="padding:5px 8px;color:{pat_color};font-size:12px">{patience}</td>
  <td style="padding:5px 8px;color:#64748b;font-family:monospace;font-size:10px">{proj[:14]}</td>
  <td style="padding:5px 8px;color:#94a3b8;font-size:11px">{ttp_str}</td>
</tr>""")
    return f"""
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;background:#0f172a">
  <thead><tr style="background:#1e293b">
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">ID</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Type</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Level</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Skill</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Status</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Rating</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Margin</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Patience</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Project</th>
    <th style="padding:7px 8px;color:#64748b;font-size:10px;text-align:left">Contract</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def build_clients_table(state: dict) -> str:
    clients = state.get("clients", {}).get("clients", [])
    if not clients:
        return "<p style='color:#64748b;padding:12px'>No clients yet.</p>"
    rows = []
    for cl in clients:
        sat = cl.get("satisfaction_score", 0)
        sat_color = "#ef4444" if sat < 0.3 else "#f59e0b" if sat < 0.6 else "#22c55e"
        bar_w = int(sat * 90)
        churn = cl.get("churn_risk", False)
        rows.append(f"""
<tr style="border-bottom:1px solid #1e293b">
  <td style="padding:7px 10px;color:#e2e8f0;font-size:12px;font-weight:600">{cl.get('client_id','?')}</td>
  <td style="padding:7px 10px;color:#94a3b8;font-size:12px">{cl.get('industry','?')}</td>
  <td style="padding:7px 10px">
    <div style="background:#1e293b;border-radius:4px;height:7px;width:90px;display:inline-block">
      <div style="background:{sat_color};height:7px;border-radius:4px;width:{bar_w}px"></div>
    </div>
    <span style="color:{sat_color};font-size:11px;margin-left:6px">{sat:.2f}</span>
  </td>
  <td style="padding:7px 10px">
    {'<span style="color:#ef4444;font-weight:700;font-size:11px">⚠ CHURN RISK</span>' if churn else '<span style="color:#22c55e;font-size:11px">✓ Active</span>'}
  </td>
  <td style="padding:7px 10px;color:#38bdf8;font-size:12px">{cl.get('num_open_projects',0)}</td>
  <td style="padding:7px 10px;color:#22c55e;font-size:12px">{cl.get('num_projects_filled',0)}</td>
  <td style="padding:7px 10px;color:#ef4444;font-size:12px">{cl.get('num_projects_expired',0)}</td>
  <td style="padding:7px 10px;color:#a78bfa;font-size:12px">${cl.get('contracted_rate',0):,.0f}/wk</td>
</tr>""")
    return f"""
<table style="width:100%;border-collapse:collapse;background:#0f172a">
  <thead><tr style="background:#1e293b">
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Client</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Industry</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Satisfaction</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Status</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Open</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Filled</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Expired</th>
    <th style="padding:7px 10px;color:#64748b;font-size:10px;text-align:left">Rate</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def build_projects_cards(state: dict) -> str:
    clients = state.get("clients", {}).get("clients", [])
    cards = []
    for cl in clients:
        for p in cl.get("projects", []):
            status = p.get("fill_status", "?")
            color = {"SEALED": "#22c55e", "PARTIAL": "#f59e0b", "OPEN": "#ef4444"}.get(status, "#94a3b8")
            deadline = p.get("deadline_remaining", "?")
            confirmed = "✓ Confirmed" if p.get("confirmed") else ""
            roles_html = ""
            for r in p.get("roles", []):
                filled = r.get("filled_count", 0)
                needed = r.get("headcount", 1)
                pct = filled / needed if needed else 0
                bar = (f'<div style="background:#334155;border-radius:2px;height:4px;margin-top:2px">'
                       f'<div style="background:#38bdf8;height:4px;border-radius:2px;width:{int(pct*80)}px"></div></div>')
                roles_html += (f'<div style="font-size:10px;color:#94a3b8;margin-top:4px">'
                               f'{r.get("developer_type","?")} · {r.get("seniority","?")} '
                               f'· {filled}/{needed}{bar}</div>')
            cards.append(f"""
<div style="background:#1e293b;border-radius:10px;padding:12px;margin:5px;flex:0 0 210px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:#e2e8f0;font-size:11px;font-weight:600;font-family:monospace">{p.get('project_id','?')[:16]}</span>
    <span style="color:{color};background:{color}22;padding:2px 5px;border-radius:4px;font-size:9px">{status}</span>
  </div>
  <div style="color:#64748b;font-size:10px;margin-top:3px">⏱ {deadline} wks  <span style="color:#22c55e">{confirmed}</span></div>
  {roles_html}
</div>""")
    if not cards:
        return "<p style='color:#64748b;padding:12px'>No active projects.</p>"
    return f'<div style="display:flex;flex-wrap:wrap">{" ".join(cards)}</div>'


def build_agent_action_history() -> str:
    """Recent agent actions as a table."""
    if not _agent_actions:
        return "<p style='color:#64748b;padding:12px'>No agent actions recorded yet.</p>"
    rows = []
    for step, tool, success, reward, result_snippet in list(_agent_actions)[-40:]:
        r_color = "#22c55e" if reward > 0 else "#ef4444" if reward < 0 else "#94a3b8"
        ok_icon = "✓" if success else "✗"
        ok_color = "#22c55e" if success else "#ef4444"
        rows.append(f"""
<tr style="border-bottom:1px solid #0f172a">
  <td style="padding:4px 8px;color:#a78bfa;font-size:11px;font-family:monospace">{step}</td>
  <td style="padding:4px 8px;color:#38bdf8;font-size:11px;font-family:monospace">{tool}</td>
  <td style="padding:4px 8px;color:{ok_color};font-size:11px">{ok_icon}</td>
  <td style="padding:4px 8px;color:{r_color};font-size:11px">{reward:+.2f}</td>
  <td style="padding:4px 8px;color:#64748b;font-size:10px;font-family:monospace">{result_snippet[:60]}</td>
</tr>""")
    return f"""
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;background:#0f172a">
  <thead><tr style="background:#1e293b">
    <th style="padding:5px 8px;color:#64748b;font-size:10px;text-align:left">Step</th>
    <th style="padding:5px 8px;color:#64748b;font-size:10px;text-align:left">Tool</th>
    <th style="padding:5px 8px;color:#64748b;font-size:10px;text-align:left">OK</th>
    <th style="padding:5px 8px;color:#64748b;font-size:10px;text-align:left">Reward</th>
    <th style="padding:5px 8px;color:#64748b;font-size:10px;text-align:left">Result</th>
  </tr></thead>
  <tbody>{''.join(reversed(rows))}</tbody>
</table>
</div>"""


def build_episode_history_table() -> str:
    """Summary of completed episodes."""
    if not _episode_stats:
        return "<p style='color:#64748b;padding:12px'>No completed episodes yet.</p>"
    rows = []
    for ep in _episode_stats[-20:]:
        profit = ep.get("final_profit", 0)
        p_color = "#22c55e" if profit >= 0 else "#ef4444"
        rows.append(f"""
<tr style="border-bottom:1px solid #0f172a">
  <td style="padding:5px 10px;color:#a78bfa;font-size:12px">{ep.get('episode', '?')}</td>
  <td style="padding:5px 10px;color:#38bdf8;font-size:12px">{ep.get('steps', '?')}</td>
  <td style="padding:5px 10px;color:{p_color};font-size:12px">${profit:,.0f}</td>
  <td style="padding:5px 10px;color:#22c55e;font-size:12px">{ep.get('placed', 0)}</td>
  <td style="padding:5px 10px;color:#f59e0b;font-size:12px">{ep.get('churned', 0)}</td>
  <td style="padding:5px 10px;color:#22c55e;font-size:12px">{ep.get('cum_reward', 0.0):+.1f}</td>
  <td style="padding:5px 10px;color:#94a3b8;font-size:11px">{ep.get('end_reason', '')}</td>
</tr>""")
    return f"""
<table style="width:100%;border-collapse:collapse;background:#0f172a">
  <thead><tr style="background:#1e293b">
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Episode</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Steps</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Profit</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Placed</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Churned</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">Cum Reward</th>
    <th style="padding:6px 10px;color:#64748b;font-size:10px;text-align:left">End Reason</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def build_candidate_lifecycle() -> str:
    """Candidate lifecycle event log."""
    if not _candidate_lifecycle:
        return "<p style='color:#64748b;padding:12px'>No lifecycle events yet.</p>"
    lines = ""
    for evt in list(_candidate_lifecycle)[-50:]:
        icon = {"hired": "✦", "placed": "▶", "left": "✗", "contract_done": "✓",
                "interview": "◎"}.get(evt.get("type", ""), "·")
        color = {"hired": "#f59e0b", "placed": "#22c55e", "left": "#ef4444",
                 "contract_done": "#38bdf8", "interview": "#a78bfa"}.get(evt.get("type", ""), "#94a3b8")
        lines += (f'<div style="font-size:11px;color:{color};font-family:monospace;padding:1px 0">'
                  f'[step {evt.get("step","?")}] {icon} {evt.get("type","?")} · '
                  f'{evt.get("candidate_id","?")[:16]} · {evt.get("detail","")}</div>\n')
    return f'<div style="background:#0a0f1a;border-radius:8px;padding:10px;height:240px;overflow-y:auto">{lines}</div>'


def build_log_panel() -> str:
    lines = list(_log_lines)[-30:]
    html = ""
    for line in reversed(lines):
        color = "#ef4444" if "ERROR" in line or "FAIL" in line else \
                "#22c55e" if ("OK" in line or "✓" in line or "sealed" in line.lower()) else \
                "#94a3b8"
        html += f'<div style="font-size:11px;color:{color};font-family:monospace;padding:1px 0">{line}</div>'
    return f'<div style="background:#0a0f1a;border-radius:8px;padding:10px;height:200px;overflow-y:auto">{html}</div>'


def _log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _log_lines.append(f"[{ts}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Main refresh
# ─────────────────────────────────────────────────────────────────────────────

_OFFLINE_HTML = "<div style='color:#ef4444;padding:20px'>⚠ Server not reachable. Start with: <code>uvicorn server.app:app --port 8000</code></div>"

def refresh():
    if not health_check():
        return (
            _OFFLINE_HTML,
            *[gr.update()] * 8,
            _OFFLINE_HTML, _OFFLINE_HTML, _OFFLINE_HTML,
            "<div style='color:#ef4444'>Server offline</div>",
            "<div style='color:#ef4444'>Server offline</div>",
            build_log_panel(),
        )

    state = fetch_full_state()
    a = state.get("agency", {})
    demand = state.get("demand", {}).get("demand_by_type", {})
    step = a.get("step", 0)

    _log(f"Step {step} | Cash ${a.get('cash_balance',0):,.0f} | Placed {a.get('num_candidates_placed',0)}")

    return (
        build_kpi_row(state),
        chart_financials(),
        chart_candidates(),
        chart_satisfaction(),
        chart_burn_runway(),
        chart_market_demand(demand),
        chart_placement_rate(),
        chart_reward_history(),
        chart_tool_usage(),
        build_candidates_table(state),
        build_clients_table(state),
        build_projects_cards(state),
        build_agent_action_history(),
        build_episode_history_table(),
        build_log_panel(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Manual tool call + overrides
# ─────────────────────────────────────────────────────────────────────────────

def manual_tool_call(tool_name: str, params_json: str) -> tuple[str, str]:
    global _current_episode
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}", build_log_panel()

    _log(f"Manual call: {tool_name}({params})")
    result = call_tool(tool_name, params)

    obs = result.get("observation", {})
    meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    reward = meta.get("reward") or result.get("reward", 0.0)
    tr = meta.get("tool_result", result)
    step = meta.get("step", len(_agent_actions))

    success = not (isinstance(tr, dict) and "error" in tr)
    snippet = str(tr)[:120] if tr else ""
    _agent_actions.append((step, tool_name, success, float(reward or 0), snippet))
    _tool_counts[tool_name] += 1
    _tool_rewards[tool_name].append(float(reward or 0))

    output = json.dumps(tr, indent=2, default=str) if isinstance(tr, dict) else str(tr)
    _log(f"  → {output[:120]}")
    return f"✅ {tool_name}\n\n{output}", build_log_panel()


def override_cash(new_cash: float) -> tuple[str, str]:
    result = call_tool("_override_cash", {"amount": new_cash})
    tr = _unwrap(result)
    msg = f"Cash set to ${new_cash:,.0f}" if not (isinstance(tr, dict) and "error" in tr) else f"Error: {tr}"
    _log(msg)
    return msg, build_log_panel()


def override_satisfaction(client_id: str, new_score: float) -> tuple[str, str]:
    result = call_tool("_override_satisfaction", {"client_id": client_id, "score": new_score})
    tr = _unwrap(result)
    msg = f"Satisfaction for {client_id} → {new_score:.2f}" if not (isinstance(tr, dict) and "error" in tr) else f"Error: {tr}"
    _log(msg)
    return msg, build_log_panel()


def do_reset(seed_str: str) -> tuple[str, str]:
    global _current_episode, _episode_start_cash
    # Save episode summary before reset
    if _last_full_state:
        a = _last_full_state.get("agency", {})
        clients = _last_full_state.get("clients", {}).get("clients", [])
        _episode_stats.append({
            "episode": _current_episode,
            "steps": a.get("step", 0),
            "final_profit": a.get("current_profit", 0),
            "placed": a.get("num_candidates_placed", 0),
            "churned": sum(1 for c in clients if c.get("churn_risk")),
            "cum_reward": sum(ac[3] for ac in _agent_actions),
            "end_reason": "done" if a.get("step", 0) >= 52 else "manual_reset",
        })
    _current_episode += 1
    seed = int(seed_str) if seed_str.strip().isdigit() else None
    reset_env(seed)
    for q in _history.values():
        q.clear()
    _agent_actions.clear()
    _tool_counts.clear()
    _tool_rewards.clear()
    _candidate_lifecycle.clear()
    _log(f"Episode {_current_episode} reset (seed={seed})")
    return f"✅ Episode {_current_episode} started (seed={seed})", build_log_panel()


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
body, .gradio-container { background: #0f172a !important; color: #e2e8f0 !important; }
.gr-box, .gr-panel { background: #1e293b !important; border: 1px solid #334155 !important; }
label, .gr-form label { color: #94a3b8 !important; }
button.primary { background: #3b82f6 !important; }
.tab-nav button { color: #94a3b8 !important; }
.tab-nav button.selected { color: #e2e8f0 !important; border-bottom: 2px solid #3b82f6 !important; }
textarea, input[type=text], input[type=number] {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; }
select { background: #0f172a !important; color: #e2e8f0 !important; }
"""

TOOL_LIST = [
    "get_agency_state", "get_client_state", "get_candidate_state",
    "get_project_details", "get_candidate_profile", "get_market_demand",
    "get_financial_summary", "find_available_projects", "confirm_project",
    "find_candidate", "interview_candidate", "hire_candidate",
    "negotiate_salary", "match_candidate_to_project", "let_go_candidate",
    "request_project_extension", "pass_on_project",
]

TOOL_PARAM_HINTS = {
    "get_client_state":           '{"client_id": "CL-FIN-00"}',
    "get_project_details":        '{"project_id": "P-CL-..."}',
    "get_candidate_profile":      '{"candidate_id": "C-BA-..."}',
    "confirm_project":            '{"project_id": "P-CL-..."}',
    "find_candidate":             '{"developer_type": "backend"}',
    "interview_candidate":        '{"candidate_id": "C-BA-..."}',
    "hire_candidate":             '{"candidate_id": "C-BA-..."}',
    "negotiate_salary":           '{"candidate_id": "C-BA-...", "offer_weekly": 1800}',
    "match_candidate_to_project": '{"candidate_id": "C-BA-...", "project_id": "P-...", "role_id": "R-..."}',
    "let_go_candidate":           '{"candidate_id": "C-BA-..."}',
    "request_project_extension":  '{"project_id": "P-CL-..."}',
    "pass_on_project":            '{"project_id": "P-CL-..."}',
}


def build_ui():
    with gr.Blocks(css=DARK_CSS, title="Staffing Agency RL Dashboard") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
<div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:22px 32px;border-radius:12px;margin-bottom:14px">
  <h1 style="color:#e2e8f0;margin:0;font-size:24px">🏢 Staffing Agency RL — Live Training Dashboard</h1>
  <p style="color:#94a3b8;margin:5px 0 0;font-size:13px">Real-time simulation viewer · Agent analytics · Manual override · Episode control</p>
</div>""")

        kpi_html = gr.HTML("<div style='color:#64748b'>Loading…</div>")

        with gr.Tabs():

            # ── Tab 1: Live Charts ─────────────────────────────────────────
            with gr.TabItem("📊 Live Charts"):
                with gr.Row():
                    chart_fin  = gr.Plot(label="Financials")
                    chart_sat  = gr.Plot(label="Client Satisfaction")
                with gr.Row():
                    chart_cand = gr.Plot(label="Candidate Pipeline")
                    chart_brn  = gr.Plot(label="Burn & Runway")
                with gr.Row():
                    chart_dem  = gr.Plot(label="Market Demand")
                    chart_plc  = gr.Plot(label="Placement Rate & Demand")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Auto-refresh")
                        refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")
                        auto_timer  = gr.Timer(value=_poll_interval)

            # ── Tab 2: Agent Analytics ─────────────────────────────────────
            with gr.TabItem("🤖 Agent Analytics"):
                gr.Markdown("""
### Agent Performance Metrics
Track what the agent is doing, how often, and the reward it earns per action.
These metrics persist across the current episode and reset on episode reset.
""")
                with gr.Row():
                    chart_rwd  = gr.Plot(label="Reward History")
                    chart_freq = gr.Plot(label="Tool Call Frequency")
                with gr.Row():
                    chart_ravg = gr.Plot(label="Avg Reward per Tool")

                gr.Markdown("### Recent Agent Actions")
                agent_actions_html = gr.HTML("<div style='color:#64748b'>No actions yet.</div>")

                gr.Markdown("### Episode History")
                episode_hist_html  = gr.HTML("<div style='color:#64748b'>No completed episodes.</div>")

                gr.Markdown("### Candidate Lifecycle Events")
                lifecycle_html     = gr.HTML("<div style='color:#64748b'>No lifecycle events yet.</div>")

            # ── Tab 3: Candidates ──────────────────────────────────────────
            with gr.TabItem("👥 Candidates"):
                cand_html = gr.HTML("<div style='color:#64748b'>Loading…</div>")

            # ── Tab 4: Clients & Projects ──────────────────────────────────
            with gr.TabItem("🏗 Clients & Projects"):
                clients_html  = gr.HTML("<div style='color:#64748b'>Loading…</div>")
                gr.Markdown("### Active Projects")
                projects_html = gr.HTML("<div style='color:#64748b'>Loading…</div>")

            # ── Tab 5: Manual Tool Call ────────────────────────────────────
            with gr.TabItem("🔧 Manual Override"):
                gr.Markdown("""
### Manually call any environment tool
Use this to inspect state, force actions, or test specific scenarios mid-episode.
Actions called here are tracked in Agent Analytics.
""")
                with gr.Row():
                    with gr.Column(scale=2):
                        tool_dropdown = gr.Dropdown(
                            choices=TOOL_LIST, value="get_agency_state", label="Tool",
                        )
                        params_box = gr.Textbox(
                            value="{}", label="Params (JSON)", lines=4,
                            placeholder='{"candidate_id": "C-BA-abc123"}',
                        )
                        call_btn = gr.Button("▶ Execute Tool", variant="primary")
                    with gr.Column(scale=3):
                        tool_output = gr.Textbox(label="Result", lines=20, interactive=False)

                tool_dropdown.change(
                    fn=lambda t: TOOL_PARAM_HINTS.get(t, "{}"),
                    inputs=tool_dropdown, outputs=params_box,
                )

                with gr.Accordion("⚠ State Override Tools", open=False):
                    gr.Markdown("Directly modify environment state. Changes are instant and do NOT advance the episode step.")
                    with gr.Row():
                        cash_input = gr.Number(value=50000, label="Set Cash Balance ($)", minimum=0)
                        cash_btn   = gr.Button("💰 Override Cash")
                        cash_msg   = gr.Textbox(label="Result", interactive=False)
                    with gr.Row():
                        sat_client = gr.Textbox(label="Client ID", placeholder="CL-FIN-00")
                        sat_score  = gr.Slider(0, 1, value=0.75, step=0.05, label="Satisfaction Score")
                        sat_btn    = gr.Button("😊 Override Satisfaction")
                        sat_msg    = gr.Textbox(label="Result", interactive=False)

            # ── Tab 6: Episode Control ─────────────────────────────────────
            with gr.TabItem("⚙ Episode Control"):
                gr.Markdown("### Reset the episode")
                with gr.Row():
                    seed_box  = gr.Textbox(value="42", label="Random Seed (blank = random)")
                    reset_btn = gr.Button("🔁 Reset Episode", variant="secondary")
                    reset_msg = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Quick actions")
                with gr.Row():
                    gr.Button("📋 List All Open Projects").click(
                        fn=lambda: manual_tool_call("find_available_projects", "{}"),
                        outputs=[tool_output, gr.HTML()],
                    )
                    gr.Button("🔍 Search All Candidates").click(
                        fn=lambda: manual_tool_call("find_candidate", "{}"),
                        outputs=[tool_output, gr.HTML()],
                    )

                gr.Markdown("### Event Log")
                log_html2 = gr.HTML(build_log_panel())

        gr.Markdown("---")
        log_html = gr.HTML(build_log_panel())

        # ─────────────────────────────────────────────────────────────────────
        # Wire outputs
        # ─────────────────────────────────────────────────────────────────────
        all_outputs = [
            kpi_html,
            chart_fin, chart_cand, chart_sat, chart_brn, chart_dem, chart_plc,
            chart_rwd, chart_freq,
            cand_html, clients_html, projects_html,
            agent_actions_html, episode_hist_html,
            log_html,
        ]

        refresh_btn.click(fn=refresh, outputs=all_outputs)
        auto_timer.tick(fn=refresh, outputs=all_outputs)

        call_btn.click(
            fn=manual_tool_call,
            inputs=[tool_dropdown, params_box],
            outputs=[tool_output, log_html],
        )
        cash_btn.click(fn=override_cash, inputs=[cash_input], outputs=[cash_msg, log_html])
        sat_btn.click(fn=override_satisfaction, inputs=[sat_client, sat_score], outputs=[sat_msg, log_html])
        reset_btn.click(fn=do_reset, inputs=[seed_box], outputs=[reset_msg, log_html])

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",         default="0.0.0.0")
    p.add_argument("--port",         type=int, default=7860)
    p.add_argument("--server_url",   default="http://localhost:8000")
    p.add_argument("--start_server", action="store_true",
                   help="Auto-start the env server before launching UI")
    p.add_argument("--poll",         type=float, default=2.0,
                   help="Auto-refresh interval in seconds")
    args = p.parse_args()

    _server_url    = args.server_url
    _poll_interval = args.poll

    if args.start_server:
        print("Starting env server on :8000 …")
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(Path(__file__).parent.parent),
        )
        time.sleep(3)
        print("Env server started.")

    print(f"\n🏢 Staffing Agency Dashboard → http://{args.host}:{args.port}")
    print(f"   Connecting to env server: {_server_url}\n")

    build_ui().launch(server_name=args.host, server_port=args.port, share=False)
