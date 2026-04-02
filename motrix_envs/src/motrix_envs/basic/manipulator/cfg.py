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

bring_ball_model_file = os.path.join(os.path.dirname(__file__), "manipulator_bring_ball.xml")


@registry.envcfg("dm-manipulator-bring-ball")
@dataclass
class BringBallCfg(EnvCfg):
    # Simulation
    model_file: str = bring_ball_model_file
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    render_spacing: float = 2.5

    # Reset sampling (match dm_control defaults).
    p_in_hand: float = 0.1
    p_in_target: float = 0.1
    randomize_arm: bool = True

    # Target sampling.
    target_x_range: tuple[float, float] = (-0.4, 0.4)
    target_z_range: tuple[float, float] = (0.1, 0.4)
    target_y: float = 0.001
    target_angle_range: tuple[float, float] = (-3.14159265, 3.14159265)

    # Object sampling.
    object_x_range: tuple[float, float] = (-0.4, 0.4)
    object_z_range: tuple[float, float] = (0.0, 0.7)
    object_angle_range: tuple[float, float] = (0.0, 6.28318531)
    object_x_vel_range: tuple[float, float] = (-5.0, 5.0)
    min_object_hand_dist: float = 0.08

    # Physics settling at episode start (in control steps, i.e. ctrl_dt units).
    # Internally this will be converted to `settle_steps * sim_substeps` physics steps.
    settle_steps: int = 80
    settle_zero_vel: bool = True

    # BringBall reward shaping.
    lift_height_threshold: float = 0.04
    touch_threshold: float = 0.01
    side_penalty_scale: float = 0.05
    side_penalty_tanh_scale: float = 10.0
    hover_penalty_scale: float = 0.02
    hover_close_threshold: float = 0.1
    post_grasp_discount: float = 0.7
    lift_height_weight: float = 0.3
    transport_weight: float = 0.7
    transport_progress_scale: float = 0.0
    transport_progress_clip: float = 0.02
    precision_weight: float = 0.0
    precision_margin: float = 0.02
    precision_value_at_margin: float = 0.1

    # BringBall-specific overrides.
    settle_steps: int = 300
    p_in_hand: float = 0.0
    p_in_target: float = 0.0
    randomize_arm: bool = False
    object_z_range: tuple[float, float] = (0.2, 0.7)
    object_x_vel_range: tuple[float, float] = (0.0, 0.0)
    hover_penalty_scale: float = 0.03
    post_grasp_discount: float = 0.0
    lift_height_weight: float = 0.1
    transport_weight: float = 2.0
    transport_progress_scale: float = 2.0
    transport_progress_clip: float = 0.02
    precision_weight: float = 1.0
    precision_margin: float = 0.01

    # Reward component weights (total reward mixing).
    reach_weight: float = 1.0
    orient_weight: float = 1.5
    pause_weight: float = 0.5
    close_weight: float = 2.0
    lift_reward_weight: float = 6.0
