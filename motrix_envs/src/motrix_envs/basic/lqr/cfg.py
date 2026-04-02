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

from motrix_envs import registry
from motrix_envs.base import EnvCfg

_DIR = os.path.dirname(__file__)


@dataclass
class LqrBaseCfg(EnvCfg):
    sim_dt: float = 0.01
    ctrl_dt: float = 0.03
    max_episode_seconds: float = None
    control_cost_coef: float = 0.1
    velocity_cost_coef: float = 0.05
    reset_position_norm: float = 2.0**0.5
    boundary_position_limit: float = 1.2
    boundary_velocity_limit: float = 8.0
    success_position_tol: float = 0.06
    success_velocity_tol: float = 0.05
    success_bonus: float = 3.0
    out_of_bounds_penalty: float = 2.0
    expected_nq: int = 0
    expected_nu: int = 0


@registry.envcfg("dm-lqr-2-1")
@dataclass
class Lqr21Cfg(LqrBaseCfg):
    model_file: str = os.path.join(_DIR, "lqr_2_1.xml")
    reset_position_norm: float = 0.8
    control_cost_coef: float = 0.15
    velocity_cost_coef: float = 0.15
    boundary_position_limit: float = 1.15
    boundary_velocity_limit: float = 6.0
    success_position_tol: float = 0.04
    success_velocity_tol: float = 0.03
    success_bonus: float = 4.0
    out_of_bounds_penalty: float = 3.0
    expected_nq: int = 2
    expected_nu: int = 1


@registry.envcfg("dm-lqr-6-2")
@dataclass
class Lqr62Cfg(LqrBaseCfg):
    model_file: str = os.path.join(_DIR, "lqr_6_2.xml")
    reset_position_norm: float = 1.0
    velocity_cost_coef: float = 0.08
    boundary_position_limit: float = 1.2
    boundary_velocity_limit: float = 8.0
    success_position_tol: float = 0.1
    success_velocity_tol: float = 0.06
    success_bonus: float = 5.0
    out_of_bounds_penalty: float = 3.0
    expected_nq: int = 6
    expected_nu: int = 2
