# OpenEnv — Core Reference

---

## What is OpenEnv?

An **end-to-end framework** for creating, deploying, and training with isolated execution environments for agentic reinforcement learning. Built on **Gymnasium-style APIs** that work identically across training, evaluation, and production.

```bash
pip install openenv-core
openenv init my_env
openenv serve my_env
openenv push
```

---

## Core Concepts

| Term | What it is |
|---|---|
| **Agent** | The learner (LLM). Takes observations, outputs actions |
| **Environment** | The world. Receives actions, updates state, returns observation + reward + done |
| **Action** | What the agent does — tool calls, code execution, text. Sent via `env.step(action)` |
| **Observation** | What the agent sees after acting. Contains `reward` (scalar) and `done` (episode over?) |
| **State** | Internal environment state tracked across steps |
| **Episode** | `reset()` → multiple `step()` calls → `done=True` → repeat |

---

## Architecture

```
Training Loop (TRL, torchforge)       Agent (LLM)
        ↓                                  ↓
  Gym-like API (WebSocket)        MCP Protocol (JSON-RPC)
  reset() · step() · state()      tools/list · tools/call
        ↓                                  ↓
             FastAPI + HTTPEnvServer
                     ↓
        Environment Instance  +  Rubric System
        (your custom logic)      (reward computation)
                     ↓
        Docker Container (Isolated Sandbox)
        Local · Swarm · Kubernetes · Daytona Cloud
```

**Two modes:**
- **SIMULATION** — full API for training (`reset` + `step` available)
- **PRODUCTION** — safe API only (no `reset`/`step` exposed)

**Key invariant:** Agents can NEVER call `reset()`. If agents could undo consequences, training breaks.

---

## File Structure

```
my_env/
├── __init__.py
├── models.py              ← Action, Observation, State types
├── client.py              ← Client class (connects to server)
├── openenv.yaml           ← Manifest
├── pyproject.toml
├── server/
│   ├── __init__.py
│   ├── my_env_environment.py   ← Core environment logic
│   ├── app.py                  ← Server entry point
│   ├── requirements.txt
│   └── Dockerfile
└── uv.lock
```

Scaffold with:
```bash
openenv init my_env
```

---

## models.py — Data Types

```python
from openenv.core.env_server.types import Action, Observation, State

class MyAction(Action):
    command: str
    target: str = ""

class MyObservation(Observation):
    output: str = ""
    flags_found: list[str] = []

class MyState(State):
    step_count: int = 0
    total_flags: int = 5
```

- **Action** — what the agent sends in
- **Observation** — what the agent gets back (always includes `reward` and `done`)
- **State** — internal tracking, never seen by agent directly

---

## server/my_environment.py — Core Logic

```python
from openenv.core.env_server.mcp_environment import MCPEnvironment
from fastmcp import FastMCP
from ..models import MyAction, MyObservation, MyState

class MyEnvironment(MCPEnvironment):
    def __init__(self):
        self.mcp = FastMCP("my_env")

        # Define MCP tools the agent can call
        @self.mcp.tool
        def scan_port(host: str, port: int) -> str:
            if port == 8080:
                return "OPEN - HTTP API (no auth)"
            return f"Port {port}: closed"

        @self.mcp.tool
        def check_sql(url: str, payload: str) -> str:
            if "' OR 1=1" in payload:
                return "VULNERABLE: SQL injection!"
            return "No vulnerability found"

        super().__init__(self.mcp)

    def reset(self, seed=None, episode_id=None, **kw) -> MyObservation:
        self._state = MyState(step_count=0, total_flags=5)
        return MyObservation(
            output="Target system ready. Find 5 flags.",
            done=False,
            reward=0.0
        )

    def step(self, action, **kw) -> MyObservation:
        self._state.step_count += 1
        result = super().step(action, **kw)   # executes MCP tool call

        # Compute reward — environment decides, never the agent
        reward = 0.2
        done = self._state.step_count >= 10

        return MyObservation(
            output=result.output,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> MyState:
        return self._state
```

**Key patterns:**
- MCP tools are defined in `__init__` with `@self.mcp.tool`
- Environment computes rewards internally — never the agent
- State tracks episode progress
- `super().step()` executes the actual tool call

---

## server/app.py — Entry Point

```python
from openenv.core.env_server import create_app
from .my_env_environment import MyEnvironment
from ..models import MyAction, MyObservation

# Pass class (factory), NOT instance
# Each WebSocket session gets its own environment instance
app = create_app(MyEnvironment, MyAction, MyObservation, env_name="my_env")
```

---

## client.py — Client Class

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import MyAction, MyObservation, MyState

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):

    # Required for async with MyEnv(...) as env:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    def _step_payload(self, action: MyAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs = MyObservation(
            output=payload.get("output", ""),
            flags_found=payload.get("flags_found", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(
            step_count=payload.get("step_count", 0),
            total_flags=payload.get("total_flags", 5),
        )
```

**Using the client:**
```python
import asyncio
from my_env.client import MyEnv
from my_env.models import MyAction

async def main():
    async with MyEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        print(obs.output)

        result = await env.step(MyAction(command="scan", target="localhost"))
        print(f"reward={result.reward}  done={result.done}")

        state = await env.state()
        print(f"steps={state.step_count}")

asyncio.run(main())
```

---

## Server Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `/ws` | WebSocket | Persistent session (used by client) |
| `/health` | HTTP GET | Health check — `{"status": "healthy"}` |
| `/reset` | HTTP POST | Reset environment (stateless) |
| `/step` | HTTP POST | Execute action (stateless) |
| `/state` | HTTP GET | Get current state |
| `/docs` | HTTP GET | OpenAPI documentation |
| `/web` | HTTP GET | Interactive web UI |

---

## Running Locally

```bash
# Option 1 — uv (recommended, hot reload)
uv run server

# Option 2 — uvicorn directly
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000 --reload

# Option 3 — multi-worker
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

# Check it's running
curl http://localhost:8000/health
```

---

## Docker

### Dockerfile — Standard Pattern
```dockerfile
# Stage 1: Build
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app
COPY . /app/env
RUN uv sync --frozen --no-editable

# Stage 2: Runtime
FROM ${BASE_IMAGE}
COPY --from=builder /app/env /app/env
HEALTHCHECK CMD python -c "..."
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
# Using OpenEnv CLI (recommended)
openenv build
openenv serve my_env

# Or with Docker directly
docker build -t my-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 my-env:latest

# With environment variables
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-env:latest
```

---

## Deployment

### HuggingFace Spaces (free, public URL, auto-scaling)
```bash
openenv push                                      # to your namespace
openenv push --repo-id username/my-env            # specific repo
openenv push --repo-id username/my-env --private  # private
```

Every HF Space gives you 3 things:

| Component | Access | Used as |
|---|---|---|
| Server | `https://<user>-<name>.hf.space` | Live API endpoint |
| Repository | `pip install git+https://huggingface.co/spaces/<user>-<name>` | Client package |
| Registry | `docker pull registry.hf.space/<user>-<name>:latest` | Docker image |

### Cloud Sandboxes
```python
# Daytona — fully isolated VMs
from openenv.containers.runtime import DaytonaProvider
provider = DaytonaProvider(api_key="...")
url = provider.start_container("my-env:latest")

# Docker Swarm — replicated with load balancing
from openenv.containers.runtime import DockerSwarmProvider
provider = DockerSwarmProvider()
url = provider.start_container("my-env:latest", replicas=4,
                                cpu_limit="2.0", memory_limit="4g")
```

---

## 6 Ways to Connect

```python
# 1. Local server (simplest)
async with MyEnv(base_url="http://localhost:8000") as env: ...

# 2. Docker image (auto-start, auto-cleanup)
env = await MyEnv.from_docker_image("my-env:latest")

# 3. HuggingFace Hub (auto-pull Docker image)
env = await MyEnv.from_env("openenv/my-env")

# 4. HF Hub without Docker (clones + runs locally)
env = await MyEnv.from_env("openenv/my-env", use_docker=False)

# 5. UV Provider (no Docker, hot reload for dev)
from openenv.containers.runtime import UVProvider
provider = UVProvider(project_path="./my_env", reload=True)
url = provider.start(port=8000)
env = MyEnv(base_url=url, provider=provider)

# 6. Docker Swarm (scaling)
from openenv.containers.runtime import DockerSwarmProvider
provider = DockerSwarmProvider()
url = provider.start_container("my-env:latest", replicas=4)
```

All methods use the same API once connected — `reset()`, `step()`, `state()`, `close()`.

---

## Scaling

### Concurrent Sessions (one server, multiple agents)
```python
app = create_app(
    MyEnv,
    max_concurrent_envs=4,
    session_timeout=300,
)
# Requires: SUPPORTS_CONCURRENT_SESSIONS = True in your env class
```

### Docker Swarm (horizontal scaling)
```python
provider = DockerSwarmProvider()
url = provider.start_container(
    "my-env:latest",
    replicas=4,
    cpu_limit="2.0",
    memory_limit="4g"
)
```

### Cloud Batch (64+ parallel episodes)
```python
results = await asyncio.gather(
    *(run_episode(i) for i in range(64))
)
# Each episode hits a separate sandbox — 64 in parallel
```

**Key principle:** One environment = one trajectory. Stack instances to generate batches. Use `EnvPool` to orchestrate collection across the stack.

---

## Available Environments (16+)

| Environment | Description |
|---|---|
| **OpenSpiel** | 70+ games — chess, poker, blackjack, Go |
| **Coding Env** | Python code execution with smolagents |
| **BrowserGym** | Browser automation — navigate, click, scrape |
| **Atari** | Breakout, Pong, Space Invaders via Gymnasium |
| **WebSearch** | Query, browse, extract from web |
| **FinRL** | Stock trading, portfolio management |
| **CARLA** | Autonomous driving simulation |
| **Git Env** | Commits, branches, merges, conflict resolution |
| **SUMO RL** | Traffic signal control, urban optimization |
| **Reasoning Gym** | Logic puzzles, math, reasoning challenges |
| **Chat Env** | Conversational dialogue training |
| **Snake / Connect4** | Classic games for quick prototyping |

All use the same 4-method API: `reset()` · `step()` · `state()` · `close()`

---

## CLI Reference

```bash
pip install openenv-core     # install
openenv init my_env          # scaffold new environment (creates 11 files)
openenv build                # build Docker image
openenv serve my_env         # build + run locally
openenv push                 # deploy to HuggingFace Spaces
openenv push --repo-id u/n   # deploy to specific repo
openenv push --private       # deploy as private space
```

---

## Key Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/meta-pytorch/OpenEnv |
| Documentation | https://meta-pytorch.org/OpenEnv/ |
| HuggingFace org | https://huggingface.co/openenv |
| HF Spaces | https://huggingface.co/openenv/spaces |
| Tutorials | https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial |
| Env examples | https://github.com/meta-pytorch/OpenEnv/tree/main/envs |