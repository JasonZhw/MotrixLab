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

"""Configuration for Microduck flat-terrain velocity training."""

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.base import EnvCfg

ENV_NAME = "microduck-flat-terrain-walk"
MODEL_FILE = str(Path(__file__).parent / "xmls" / "scene_walk.xml")

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

# The STAND keyframe from pollen-robotics/microduck_rl. Keep this order aligned
# with JOINT_NAMES and the 14-dimensional runtime action contract.
DEFAULT_JOINT_ANGLES = (
    0.0,
    -0.0872664626,
    -0.457924,
    -0.004940,
    0.452984,
    0.3490658504,
    0.3490658504,
    0.0,
    0.0,
    0.0,
    0.0872664626,
    0.457924,
    0.004940,
    -0.452984,
)


@dataclass
class NoiseCfg:
    enabled: bool = True
    gyro: float = 0.03
    gravity: float = 0.01
    joint_pos: float = 0.001
    joint_vel: float = 0.25


@dataclass
class ControlCfg:
    # Position target = default joint angle + action_scale * clipped action.
    # Keep this aligned with pollen-robotics/microduck_rl. A smaller 0.25-rad
    # scale was sufficient for standing but prevented a useful leg swing.
    action_scale: float = 1.0


@dataclass
class InitStateCfg:
    root_height: float = 0.12
    root_xy_noise: float = 0.005
    max_roll_deg: float = 5.0
    max_pitch_deg: float = 10.0
    joint_pos_noise: float = 0.03
    joint_vel_noise: float = 0.10
    default_joint_angles: tuple[float, ...] = DEFAULT_JOINT_ANGLES


@dataclass
class CommandCfg:
    # Shared deployment block: [twist(3), head_pose(4), body_pose(6)].
    lin_vel_x: tuple[float, float] = (-0.4, 0.4)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    head_pose: tuple[tuple[float, float], ...] = (
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.07, 0.07),
        (-0.015, 0.015),
    )
    body_pose: tuple[tuple[float, float], ...] = (
        (-0.005, 0.005),
        (-0.005, 0.005),
        (-0.005, 0.005),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
    )
    resampling_time: tuple[float, float] = (4.0, 8.0)
    zero_twist_probability: float = 0.20
    # Curriculum stages are measured in environment control steps. RSL-RL uses
    # 24 control steps per iteration, so these correspond to 300 and 700 PPO
    # iterations respectively. The first stage teaches a clean forward gait
    # before lateral motion and aggressive turning are introduced.
    forward_only_steps: int = 300 * 24
    mixed_motion_steps: int = 700 * 24


@dataclass
class NormalizationCfg:
    gyro: float = 1.0
    gravity: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 0.05
    last_action: float = 1.0


@dataclass
class AssetCfg:
    body_name: str = "trunk_base"
    ground_name: str = "floor"
    foot_names: tuple[str, str] = ("left_foot_collision", "right_foot_collision")


@dataclass
class SensorCfg:
    local_lin_vel: str = "imu_lin_vel"
    gyro: str = "imu_ang_vel"


@dataclass
class TerminationCfg:
    min_root_height: float = 0.055
    max_tilt_deg: float = 60.0


@dataclass
class RewardCfg:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "track_linear_velocity": 4.0,
            "directional_velocity": 2.0,
            "track_angular_velocity": 1.0,
            "head_pose_tracking": 0.5,
            "upright": 2.0,
            "pose": 0.3,
            "base_height": 0.5,
            "air_time": 2.0,
            "swing_time": 0.75,
            "foot_clearance": 0.75,
            "single_support": 0.50,
            "double_support": -0.25,
            "gait_balance": -0.10,
            "foot_slip": -0.20,
            "vertical_velocity": -1.0,
            "body_ang_vel": -0.05,
            "action_rate": -0.03,
            "joint_velocity": -1.0e-4,
            "joint_acceleration": -2.5e-7,
            "joint_limits": -2.0,
            "termination": -5.0,
        }
    )
    linear_tracking_std: float = 0.15
    angular_tracking_std: float = 0.7071067812  # sqrt(0.5)
    upright_std: float = 0.2236067977  # sqrt(0.05)
    head_tracking_std: float = 0.5
    standing_pose_std: float = 0.12
    walking_pose_std: float = 0.30
    target_root_height: float = 0.12
    root_height_std: float = 0.02
    air_time_min: float = 0.06
    air_time_max: float = 0.300
    target_foot_height: float = 0.020
    foot_height_std: float = 0.012


@registry.envcfg(ENV_NAME)
@dataclass
class MicroduckWalkNpEnvCfg(EnvCfg):
    """Microduck velocity-tracking task using MotrixSim's NumPy backend."""

    model_file: str = MODEL_FILE
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 20.0
    render_spacing: float = 0.5
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    init_state: InitStateCfg = field(default_factory=InitStateCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    asset: AssetCfg = field(default_factory=AssetCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)

    def validate(self):
        super().validate()
        if len(self.init_state.default_joint_angles) != 14:
            raise ValueError("Microduck default pose must contain 14 joint angles")
        if self.control.action_scale <= 0:
            raise ValueError("action_scale must be positive")
        if not 0.0 <= self.commands.zero_twist_probability <= 1.0:
            raise ValueError("zero_twist_probability must be in [0, 1]")
