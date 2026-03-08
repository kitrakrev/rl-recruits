# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import MyAction, MyObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import MyAction, MyObservation, MyState   # ← MyState must be here

class MyEnv(
    EnvClient[MyAction, MyObservation, MyState]        # ← 3 params
):
    def _step_payload(self, action: MyAction):
        return action.model_dump()

    def _parse_result(self, payload):
        obs = MyObservation(**payload)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload):
        return MyState(**payload)