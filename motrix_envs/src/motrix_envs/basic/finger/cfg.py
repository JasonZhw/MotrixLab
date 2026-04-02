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

model_file = os.path.dirname(__file__) + "/finger.xml"
spin_model_file = os.path.dirname(__file__) + "/finger_spin.xml"
turn_easy_model_file = os.path.dirname(__file__) + "/finger_turn_easy.xml"
turn_hard_model_file = os.path.dirname(__file__) + "/finger_turn_hard.xml"


@dataclass
class FingerBaseCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02

    # Task setup
    task: str = "spin"  # "spin" | "turn"
    target_radius: float = 0.07

    # Reward thresholds (match dm_control defaults)
    spin_velocity_threshold: float = 15.0

    # Reward mode
    # - "sparse": match dm_control (1 if hinge_velocity <= -threshold else 0)
    # - "shaped": dense reward to make training easier
    reward_mode: str = "sparse"
    shaped_reward_beta: float = 1.0

    # Extra shaping for Spin tasks (helps reduce "no contact" failures)
    spin_touch_bonus_scale: float = 0.0
    spin_touch_bonus_tanh_scale: float = 50.0
    spin_approach_reward_scale: float = 0.0
    spin_approach_sigma: float = 0.15
    # Turn shaping: reward falls linearly to 0 at margin = scale * target_radius
    turn_reward_margin_scale: float = 4.0
    turn_reward_min_margin: float = 0.0
    turn_shaped_reward_beta: float = 1.0
    # Turn shaping mode:
    # - "linear": clip(1 - max(dist,0)/margin, 0..1)
    # - "exp": exp(-max(dist,0)/sigma)
    turn_reward_shape: str = "linear"
    turn_reward_sigma_scale: float = 1.0
    turn_reward_sigma_min: float = 0.05
    # Extra shaping for Turn tasks (to reduce jitter and help contact)
    turn_touch_bonus_scale: float = 0.05
    turn_touch_bonus_tanh_scale: float = 50.0
    # Encourage approaching the spinner (helps avoid "no contact" deadlock)
    turn_approach_reward_scale: float = 0.3
    turn_approach_sigma: float = 0.15
    turn_action_l2_penalty_scale: float = 0.002
    turn_action_delta_l2_penalty_scale: float = 0.01

    # Reset sampling
    reset_collision_free_attempts: int = 200


@registry.envcfg("dm-finger-spin")
@dataclass
class FingerSpinCfg(FingerBaseCfg):
    model_file: str = spin_model_file
    task: str = "spin"
    reward_mode: str = "shaped"
    spin_approach_reward_scale: float = 0.15
    spin_touch_bonus_scale: float = 0.03


@registry.envcfg("dm-finger-turn-easy")
@dataclass
class FingerTurnEasyCfg(FingerBaseCfg):
    model_file: str = turn_easy_model_file
    task: str = "turn"
    target_radius: float = 0.07
    reward_mode: str = "shaped"
    turn_reward_shape: str = "exp"


@registry.envcfg("dm-finger-turn-hard")
@dataclass
class FingerTurnHardCfg(FingerTurnEasyCfg):
    model_file: str = turn_hard_model_file
    target_radius: float = 0.03
    reward_mode: str = "shaped"
    turn_reward_shape: str = "exp"
