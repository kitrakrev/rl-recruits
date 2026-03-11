"""
OpenEnv FastAPI app for the Staffing Agency environment.

Uses openenv-core's create_app() factory — exactly as echo_env does.

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Or:
    uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

Config API (live reload without restart):
    GET  /config/env         → current environment parameters
    GET  /config/training    → default training hyperparameters
    GET  /config             → both combined
    PATCH /config/env        → update environment parameters (affects new sessions)

Examples:
    curl http://localhost:8000/config/env
    curl -X PATCH http://localhost:8000/config/env \\
         -H "Content-Type: application/json" \\
         -d '{"curriculum_stage": 2, "passive_streak_threshold": 5}'
"""
import dataclasses
import json
import os
from typing import Any

from fastapi import Body, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from env.config import Config
from training.config import TrainingConfig
from server.staffing_environment import StaffingAgencyEnvironment

# ---------------------------------------------------------------------------
# Environment configuration (mutable — PATCH /config/env updates this live)
# ---------------------------------------------------------------------------
_stage = int(os.getenv("CURRICULUM_STAGE", "1"))
_default_config = Config(
    curriculum_stage=_stage,
    max_roles_per_project=1 if _stage == 1 else (2 if _stage == 2 else 3),
    t_deadline_min=8  if _stage == 1 else (6 if _stage == 2 else 4),
    t_deadline_max=14 if _stage == 1 else (10 if _stage == 2 else 10),
    num_clients=1 if _stage == 1 else 3,
    llm_mode=os.getenv("LLM_MODE", "stub"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

# ---------------------------------------------------------------------------
# Create OpenEnv app
# The factory partial passes the SAME config object reference each time a new
# environment session is created. Updating _default_config fields via PATCH
# /config/env will affect all subsequent sessions automatically.
# ---------------------------------------------------------------------------
app = create_app(
    env=lambda: StaffingAgencyEnvironment(config=_default_config),
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="staffing_agency",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Middleware: fix web UI form sending `arguments` as a JSON string instead of dict
# The HTML form always sends values as strings, but CallToolAction.arguments
# expects a dict. This middleware intercepts /web/step requests and parses
# the `arguments` string into a proper JSON object.
# ---------------------------------------------------------------------------

class _FixArgumentsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/web/step" and request.method == "POST":
            body = await request.body()
            try:
                data = json.loads(body)
                action = data.get("action", {})
                args = action.get("arguments")
                if isinstance(args, str):
                    try:
                        action["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        action["arguments"] = {}
                    data["action"] = action
                    body = json.dumps(data).encode()

                    # Rebuild request with fixed body
                    from starlette.datastructures import MutableHeaders
                    async def fixed_body():
                        return body
                    request._body = body
            except (json.JSONDecodeError, ValueError):
                pass
        return await call_next(request)

app.add_middleware(_FixArgumentsMiddleware)

# ---------------------------------------------------------------------------
# Config API — mounted directly on the FastAPI app returned by create_app
# ---------------------------------------------------------------------------

@app.get(
    "/config/env",
    summary="Get current environment configuration",
    tags=["config"],
)
def get_env_config() -> dict:
    """Return all current environment parameters as a JSON object.

    These values control episode length, economics, curriculum stage,
    server penalties, and LLM mode. Sensitive fields (api keys) are redacted.
    """
    d = _default_config.to_dict()
    # Redact sensitive values
    if d.get("anthropic_api_key"):
        d["anthropic_api_key"] = "***"
    return d


@app.patch(
    "/config/env",
    summary="Update environment configuration (live)",
    tags=["config"],
)
def patch_env_config(updates: dict[str, Any] = Body(..., example={
    "curriculum_stage": 2,
    "passive_streak_threshold": 5,
    "repeat_call_penalty": -200.0,
})) -> dict:
    """Partially update environment configuration.

    Only recognised fields are applied; unknown keys are silently ignored.
    Changes take effect for all **new** environment sessions. Any currently
    active session (if one exists) is not affected mid-episode.

    Returns the updated config alongside a list of keys that were changed.

    Example — switch to curriculum stage 2:
        PATCH /config/env
        {"curriculum_stage": 2, "num_clients": 3, "max_roles_per_project": 2}
    """
    changed = _default_config.update(updates)
    if not changed:
        raise HTTPException(
            status_code=400,
            detail=f"No valid fields found. Known fields: {[f.name for f in dataclasses.fields(_default_config)]}",
        )
    d = _default_config.to_dict()
    if d.get("anthropic_api_key"):
        d["anthropic_api_key"] = "***"
    return {"updated": changed, "config": d}


@app.get(
    "/config/training",
    summary="Get default training hyperparameters",
    tags=["config"],
)
def get_training_config() -> dict:
    """Return the default TrainingConfig values.

    These are the hyperparameters used by the training loop
    (training/reinforce.py). To override them, pass CLI flags or create
    a training/config.yaml file.
    """
    cfg = TrainingConfig()
    d = cfg.to_dict()
    if d.get("wandb_api_key"):
        d["wandb_api_key"] = "***"
    return d


@app.get(
    "/config",
    summary="Get all configuration (env + training defaults)",
    tags=["config"],
)
def get_all_config() -> dict:
    """Return both environment and training configurations in one call."""
    env_d = _default_config.to_dict()
    if env_d.get("anthropic_api_key"):
        env_d["anthropic_api_key"] = "***"
    train_d = TrainingConfig().to_dict()
    if train_d.get("wandb_api_key"):
        train_d["wandb_api_key"] = "***"
    return {"env": env_d, "training_defaults": train_d}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
