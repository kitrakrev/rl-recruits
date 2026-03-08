"""
OpenEnv FastAPI app for the Staffing Agency environment.

Uses openenv-core's create_app() factory — exactly as echo_env does.

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Or:
    uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
import functools
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from env.config import Config
from server.staffing_environment import StaffingAgencyEnvironment

# More learnable default: single-role projects with longer deadlines
# so the agent can actually seal projects and see positive reward.
# curriculum_stage=1 can be overridden via CURRICULUM_STAGE env var.
import os
_stage = int(os.getenv("CURRICULUM_STAGE", "1"))
_default_config = Config(
    curriculum_stage=_stage,
    max_roles_per_project=1 if _stage == 1 else (2 if _stage == 2 else 3),
    t_deadline_min=8  if _stage == 1 else (6 if _stage == 2 else 4),
    t_deadline_max=14 if _stage == 1 else (10 if _stage == 2 else 10),
    num_clients=1 if _stage == 1 else 3,
)

app = create_app(
    env=functools.partial(StaffingAgencyEnvironment, config=_default_config),
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="staffing_agency",
    max_concurrent_envs=1,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
