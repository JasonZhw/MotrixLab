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
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    stiffness = 80  # [N*m/rad]
    damping = 1  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.05


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.42]

    # the default angles for all joints. key = joint name, value = target angle [rad]
    default_joint_angles = {
        "FL_hip": 0.0,  # [rad]
        "RL_hip": 0.0,  # [rad]
        "FR_hip": -0.0,  # [rad]
        "RR_hip": -0.0,  # [rad]
        "FL_thigh": 0.9,  # [rad]
        "RL_thigh": 0.9,  # [rad]
        "FR_thigh": 0.9,  # [rad]
        "RR_thigh": 0.9,  # [rad]
        "FL_calf": -1.8,  # [rad]
        "RL_calf": -1.8,  # [rad]
        "FR_calf": -1.8,  # [rad]
        "RR_calf": -1.8,  # [rad]
    }


@dataclass
class Commands:
    vel_limit = [
        [-1.0, -1.0, -1.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [2.0, 1.0, 1.0],  # max
    ]


@dataclass
class Normalization:
    lin_vel = 2
    ang_vel = 0.25
    dof_pos = 1
    dof_vel = 0.05


@dataclass
class Asset:
    body_name = "trunk"
    foot_name = "foot"
    ground_name = "floor"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["trunk"]


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet = ["FR", "FL", "RR", "RL"]


# -- docs-tag-start: go1-reward-config --
@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -0.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -0.0,
            "torques": -0.00001,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "base_height": -0.0,
            "feet_air_time": 1.0,
            "collision": -1.0 * 0,
            "action_rate": -0.001,
            "stand_still": -0.0,
            "hip_pos": -1,
            "calf_pos": -0.3 * 0,
        }
    )

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1


# -- docs-tag-end: go1-reward-config --


@registry.envcfg("go1-flat-terrain-walk")
@dataclass
class Go1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_motor_actuator.xml"
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01


@registry.envcfg("go1-rough-terrain-walk")
@dataclass
class Go1WalkNpRoughEnvCfg(Go1WalkNpEnvCfg):
    render_spacing: float = 0.0
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_rough_terrain.xml"


@registry.envcfg("go1-stairs-terrain-walk")
@dataclass
class Go1WalkNpStairsEnvCfg(Go1WalkNpEnvCfg):
    render_spacing: float = 0.0
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_terrain.xml"

    @dataclass
    class Commands:
        vel_limit = [
            [0.5, -0.0, 0.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
            [1.0, 0.0, 0.0],  # max
        ]

    @dataclass
    class RewardConfig:
        scales: dict[str, float] = field(
            default_factory=lambda: {
                "termination": -0.0,
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.5,
                "lin_vel_z": -2.0,
                "ang_vel_xy": -0.05,
                "orientation": -0.0,
                "torques": -0.00001,
                "dof_vel": -0.0,
                "dof_acc": -2.5e-7,
                "base_height": -0.0,
                "feet_air_time": 1.0,
                "collision": -1.0 * 0,
                "feet_stumble": -0.1,
                "action_rate": -0.001,
                "stand_still": -0.0,
                "hip_pos": -1,
                "calf_pos": -0.3 * 0,
            }
        )

        tracking_sigma: float = 0.25
        max_foot_height: float = 0.1

    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
