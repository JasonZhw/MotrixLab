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

"""Configuration for Microduck's ground-pick motion."""

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.locomotion.microduck.cfg import (
    AssetCfg,
    InitStateCfg,
    MicroduckWalkNpEnvCfg,
    RewardCfg,
    TerminationCfg,
)

ENV_NAME = "microduck-ground-pick"
MODEL_FILE = str(Path(__file__).parent / "xmls" / "scene_ground_pick.xml")


@dataclass
class GroundPickInitStateCfg(InitStateCfg):
    max_roll_deg: float = 3.0
    max_pitch_deg: float = 5.0
    joint_pos_noise: float = 0.02
    joint_vel_noise: float = 0.05


@dataclass
class GroundPickAssetCfg(AssetCfg):
    head_collision_names: tuple[str, ...] = (
        "top_head_collision",
        "jaw_collision",
        "bottom_head_collision",
    )
    mouth_site_name: str = "mouth_tip"


@dataclass
class GroundPickTerminationCfg(TerminationCfg):
    # A deep, controlled crouch is valid; lying flat is not.
    min_root_height: float = 0.035
    max_tilt_deg: float = 70.0


@dataclass
class GroundPickRewardCfg(RewardCfg):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "mouth_ground_proximity": 3.0,
            "mouth_perpendicular": 2.0,
            "return_pose_legs": 6.0,
            "return_pose_neck": 6.0,
            "return_upright": 4.0,
            "upright": 0.2,
            "feet_grounded": 3.0,
            "feet_flat": -2.0,
            "head_contact": -8.0,
            "neck_velocity_descent": -0.1,
            "horizontal_velocity": -0.5,
            "vertical_velocity": -0.5,
            "body_ang_vel": -0.05,
            "action_rate": -0.08,
            # Penalize only fast leg-command changes. Using a mean keeps this
            # gentle enough for the intended slow crouch and stand-up motion.
            "leg_action_rate": -1.0,
            "joint_velocity": -2.0e-4,
            "joint_acceleration": -5.0e-7,
            "joint_limits": -2.0,
            "termination": -8.0,
        }
    )
    period: float = 4.0
    descent_end: float = 0.375
    hold_end: float = 0.425
    rise_end: float = 0.80
    mouth_height_std: float = 0.10
    leg_return_std: float = 0.30
    neck_return_std: float = 0.15
    return_upright_std: float = 0.40
    always_upright_std: float = 0.50


@registry.envcfg(ENV_NAME)
@dataclass
class MicroduckGroundPickNpEnvCfg(MicroduckWalkNpEnvCfg):
    """Crouch, point the beak down near the floor, then stand again."""

    model_file: str = MODEL_FILE
    max_episode_seconds: float = 12.0
    init_state: GroundPickInitStateCfg = field(default_factory=GroundPickInitStateCfg)
    asset: GroundPickAssetCfg = field(default_factory=GroundPickAssetCfg)
    termination: GroundPickTerminationCfg = field(default_factory=GroundPickTerminationCfg)
    reward: GroundPickRewardCfg = field(default_factory=GroundPickRewardCfg)

    def validate(self):
        super().validate()
        reward = self.reward
        if not 0.0 < reward.descent_end < reward.hold_end < reward.rise_end < 1.0:
            raise ValueError("ground-pick phase boundaries must be strictly increasing in (0, 1)")
        if reward.period <= 0.0:
            raise ValueError("ground-pick period must be positive")
