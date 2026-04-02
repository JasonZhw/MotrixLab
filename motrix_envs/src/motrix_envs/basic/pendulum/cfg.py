# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from dataclasses import dataclass

import numpy as np

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/pendulum.xml"


# -- docs-tag-start: pendulum-env-cfg --
@registry.envcfg("pendulum")
@dataclass
class PendulumEnvCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.0125
    ctrl_dt: float = 0.025
    angle_bound: float = 8.0
    cosing_bound: float = 0.0
    # reset_noise_scale: float = 0.01
    # -- docs-tag-end: pendulum-env-cfg --

    def __post_init__(self):
        self.cosing_bound = float(np.cos(np.deg2rad(self.angle_bound)))
