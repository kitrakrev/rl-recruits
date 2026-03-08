# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is a simple test environment that echoes back messages.
"""

from pydantic import Field

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