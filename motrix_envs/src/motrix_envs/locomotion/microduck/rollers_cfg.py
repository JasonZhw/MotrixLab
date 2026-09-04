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

"""Configuration for Microduck's one-turn roller spin."""

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.locomotion.microduck.cfg import (
    AssetCfg,
    CommandCfg,
    InitStateCfg,
    MicroduckWalkNpEnvCfg,
    RewardCfg,
    TerminationCfg,
)

ENV_NAME = "microduck-flat-terrain-rollers"
MODEL_FILE = str(Path(__file__).parent / "xmls" / "scene_rollers.xml")


@dataclass
class RollerInitStateCfg(InitStateCfg):
    # The wheel radius lifts the same standing pose by about 23.5 mm.
    root_height: float = 0.1435
    joint_pos_noise: float = 0.02
    joint_vel_noise: float = 0.05


@dataclass
class RollerCommandCfg(CommandCfg):
    # The first two command slots carry cos/sin of the four-second action phase.
    # All remaining slots stay zero, preserving the shared 61D observation.
    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    head_pose: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 4
    body_pose: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 6
    zero_twist_probability: float = 0.0


@dataclass
class RollerAssetCfg(AssetCfg):
    head_height_site_name: str = "head_imu"
    foot_names: tuple[str, ...] = (
        "left_front_wheel_collision",
        "left_rear_wheel_collision",
        "right_front_wheel_collision",
        "right_rear_wheel_collision",
    )


@dataclass
class RollerTerminationCfg(TerminationCfg):
    min_root_height: float = 0.075
    max_tilt_deg: float = 65.0


@dataclass
class RollerRewardCfg(RewardCfg):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "spin_rate_track": 6.0,
            "spin_rate_l1": 0.5,
            "stay_in_place": -3.0,
            "wheel_differential": 1.0,
            "leg_antisymmetry": 1.0,
            "both_feet_grounded": 0.5,
            "upright": 3.0,
            "rest_pose": 2.0,
            "base_height": 2.0,
            "head_stability": 1.0,
            "head_height": 2.0,
            "feet_flat": -2.0,
            "vertical_velocity": -1.0,
            "body_ang_vel": -0.05,
            "action_rate": -0.05,
            "joint_velocity": -1.0e-4,
            "joint_acceleration": -2.5e-7,
            "joint_limits": -1.0,
            "termination": -5.0,
        }
    )
    # Four-second profile adapted from pollen-robotics/microduck_rl:
    # 0.5 s accelerate, 1.6 s at 3 rad/s, 0.5 s brake, 1.4 s stand.
    # The area under that profile is 6.3 rad, approximately one full turn.
    period: float = 4.0
    spin_rate_max: float = 3.0
    spin_rate_std: float = 1.5
    accel_end: float = 0.125
    hold_end: float = 0.525
    brake_end: float = 0.650
    launch_drift_scale: float = 0.2
    wheel_omega_scale: float = 17.0
    leg_antisymmetry_half_step: int = 1_500 * 24
    leg_antisymmetry_quarter_step: int = 3_000 * 24
    target_root_height: float = 0.1435
    root_height_std: float = 0.025
    head_velocity_std: float = 1.0
    min_head_clearance: float = 0.11
    head_height_std: float = 0.025
    roller_pose_std: float = 0.35


@registry.envcfg(ENV_NAME)
@dataclass
class MicroduckRollersNpEnvCfg(MicroduckWalkNpEnvCfg):
    """Spin once counter-clockwise on four passive wheels, then stand."""

    model_file: str = MODEL_FILE
    max_episode_seconds: float = 20.0
    init_state: RollerInitStateCfg = field(default_factory=RollerInitStateCfg)
    commands: RollerCommandCfg = field(default_factory=RollerCommandCfg)
    asset: RollerAssetCfg = field(default_factory=RollerAssetCfg)
    termination: RollerTerminationCfg = field(default_factory=RollerTerminationCfg)
    reward: RollerRewardCfg = field(default_factory=RollerRewardCfg)

    def validate(self):
        super().validate()
        reward = self.reward
        if reward.period <= 0.0:
            raise ValueError("spin period must be positive")
        if not 0.0 < reward.accel_end < reward.hold_end < reward.brake_end < 1.0:
            raise ValueError("spin phase boundaries must be strictly increasing in (0, 1)")
        phase_widths = (
            reward.accel_end,
            reward.hold_end - reward.accel_end,
            reward.brake_end - reward.hold_end,
            1.0 - reward.brake_end,
        )
        if reward.period * min(phase_widths) < self.ctrl_dt:
            raise ValueError("each spin phase must last at least one control step")
        if reward.spin_rate_max <= 0.0:
            raise ValueError("spin_rate_max must be positive")
        if reward.spin_rate_std <= 0.0:
            raise ValueError("spin_rate_std must be positive")
        if reward.wheel_omega_scale <= 0.0:
            raise ValueError("wheel_omega_scale must be positive")
        if not 0.0 <= reward.launch_drift_scale <= 1.0:
            raise ValueError("launch_drift_scale must be in [0, 1]")
        if self.reward.min_head_clearance <= 0.0:
            raise ValueError("min_head_clearance must be positive")
        if self.reward.head_height_std <= 0.0:
            raise ValueError("head_height_std must be positive")
        if not 0 < reward.leg_antisymmetry_half_step < reward.leg_antisymmetry_quarter_step:
            raise ValueError("leg antisymmetry curriculum steps must be increasing")
