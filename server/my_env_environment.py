# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
My Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from ..models import MyAction, MyObservation, MyState

from openenv.core.env_server.mcp_environment import MCPEnvironment
from fastmcp import FastMCP

class MyEnvironment(MCPEnvironment):
    def __init__(self):
        self.mcp = FastMCP("my_env")

        @self.mcp.tool
        def some_tool(target: str, port: int) -> str:
            # your logic here
            return f"result"

        super().__init__(self.mcp)

    def reset(self, **kw) -> MyObservation:
        self._state = MyState()
        return MyObservation(output="Ready.", done=False, reward=0.0)

    def step(self, action, **kw) -> MyObservation:
        self._state.step_count += 1
        result = super().step(action, kw)
        reward = 0.2  # your reward logic
        done = self._state.step_count >= 10
        return MyObservation(output=result.output, reward=reward, done=done)
    async def __aenter__(self):
        await self.connect()   # opens the WebSocket connection
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()     # cleans up on exit
        return False

    @property
    def state(self) -> MyState:
        return self._state