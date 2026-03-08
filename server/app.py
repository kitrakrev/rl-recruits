"""
OpenEnv FastAPI app for the Staffing Agency environment.

Uses openenv-core's create_app() factory — exactly as echo_env does.

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Or:
    uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.staffing_environment import StaffingAgencyEnvironment

app = create_app(
    env_cls=StaffingAgencyEnvironment,
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="staffing_agency",
    max_concurrent_envs=1,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
