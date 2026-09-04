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

"""Microduck roller task: spin once in place, brake, then stand."""

import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.microduck.cfg import JOINT_NAMES
from motrix_envs.locomotion.microduck.rollers_cfg import ENV_NAME, MicroduckRollersNpEnvCfg
from motrix_envs.locomotion.microduck.walk_np import (
    HEAD_SLICE,
    NUM_COMMANDS,
    MicroduckWalkTask,
)
from motrix_envs.np.env import NpEnvState

WHEEL_NAMES = (
    "passive_LF_wheel",
    "passive_LR_wheel",
    "passive_RF_wheel",
    "passive_RR_wheel",
)
SCISSOR_LEFT_INDICES = np.asarray(
    [JOINT_NAMES.index(name) for name in ("left_hip_pitch", "left_knee")],
    dtype=np.int64,
)
SCISSOR_RIGHT_INDICES = np.asarray(
    [JOINT_NAMES.index(name) for name in ("right_hip_pitch", "right_knee")],
    dtype=np.int64,
)


def spin_rate_by_phase(
    phase: np.ndarray,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> np.ndarray:
    """Return the counter-clockwise yaw-rate target for a normalized phase."""
    phase = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    target = np.zeros_like(phase)

    accelerating = phase < accel_end
    target[accelerating] = rate_max * phase[accelerating] / accel_end

    holding = np.logical_and(phase >= accel_end, phase < hold_end)
    target[holding] = rate_max

    braking = np.logical_and(phase >= hold_end, phase < brake_end)
    target[braking] = rate_max * (
        1.0 - (phase[braking] - hold_end) / (brake_end - hold_end)
    )
    return target


def spin_wheel_differential_reward(
    wheel_omega: np.ndarray,
    spin_gate: np.ndarray,
    omega_scale: float,
) -> np.ndarray:
    """Reward left wheels rolling backward and right wheels rolling forward."""
    left_omega = np.mean(wheel_omega[:, :2], axis=1)
    right_omega = np.mean(wheel_omega[:, 2:], axis=1)
    differential = np.maximum(right_omega - left_omega, 0.0)
    return spin_gate * np.tanh(differential / omega_scale)


def _encode_phase(phase: np.ndarray) -> np.ndarray:
    """Encode phase without changing the shared 13D command contract."""
    commands = np.zeros((phase.shape[0], NUM_COMMANDS), dtype=np.float32)
    angle = 2.0 * np.pi * phase
    commands[:, 0] = np.cos(angle)
    commands[:, 1] = np.sin(angle)
    return commands


@registry.env(ENV_NAME, sim_backend="np")
class MicroduckRollersTask(MicroduckWalkTask):
    """Learn a four-second roller maneuver: turn once and stop upright.

    A normalized phase is encoded in the first two command slots. Its target
    yaw-rate profile integrates to about 2*pi radians per cycle. The remaining
    command slots stay zero, so the policy retains the shared 61D observation.
    """

    def __init__(self, cfg: MicroduckRollersNpEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._head_height_site = self._model.get_site(cfg.asset.head_height_site_name)
        wheel_joints = tuple(self._model.get_joint(name) for name in WHEEL_NAMES)
        self._wheel_qvel_indices = np.asarray(
            [joint.dof_vel_index for joint in wheel_joints], dtype=np.int64
        )

    def get_wheel_vel(self, data) -> np.ndarray:
        return data.dof_vel[:, self._wheel_qvel_indices]

    def _sample_commands(self, num_envs: int) -> np.ndarray:
        # Deployment starts the gesture from a stable standing pose at phase 0.
        return _encode_phase(np.zeros(num_envs, dtype=np.float32))

    def _update_commands(self, info: dict) -> None:
        self._global_step += 1
        previous_phase = info["phase"]
        phase = previous_phase + np.float32(self.cfg.ctrl_dt / self.cfg.reward.period)
        phase %= np.float32(1.0)
        wrapped = phase < previous_phase
        if np.any(wrapped):
            info["last_cycle_angle"][wrapped] = info["spin_angle"][wrapped]
            info["spin_angle"][wrapped] = 0.0
        info["phase"] = phase
        angle = np.float32(2.0 * np.pi) * phase
        info["commands"][:, 0] = np.cos(angle)
        info["commands"][:, 1] = np.sin(angle)

    def _get_wheel_contacts(self, data) -> np.ndarray:
        contacts = self._model.get_contact_query(data).is_colliding(self._foot_contact_pairs)
        return contacts.reshape((self.num_envs, 4))

    def _get_foot_contacts(self, data) -> np.ndarray:
        wheel_contacts = self._get_wheel_contacts(data)
        return np.stack(
            (np.any(wheel_contacts[:, :2], axis=1), np.any(wheel_contacts[:, 2:], axis=1)),
            axis=1,
        )

    def reset(self, data) -> tuple[np.ndarray, dict]:
        obs, info = super().reset(data)
        num_reset = data.shape[0]
        info["phase"] = np.zeros(num_reset, dtype=np.float32)
        info["spin_angle"] = np.zeros(num_reset, dtype=np.float32)
        info["last_cycle_angle"] = np.zeros(num_reset, dtype=np.float32)
        return obs, info

    def _feet_flat_cost(self, state: NpEnvState) -> np.ndarray:
        costs = []
        for index, foot in enumerate(self._feet):
            rotation = foot.get_rotation_mat(state.data)
            local_gravity = np.einsum("nji,j->ni", rotation, self.gravity_vec)
            cost = np.sum(np.square(local_gravity[:, :2]), axis=1)
            costs.append(cost * state.info["contacts"][:, index])
        return np.sum(np.stack(costs, axis=1), axis=1)

    def _leg_antisymmetry_scale(self) -> float:
        cfg = self.cfg.reward
        if self._global_step >= cfg.leg_antisymmetry_quarter_step:
            return 0.25
        if self._global_step >= cfg.leg_antisymmetry_half_step:
            return 0.5
        return 1.0

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        values = self._reward_values(state)
        state.info["spin_angle"] += values["gyro"][:, 2] * self.cfg.ctrl_dt
        raw = self._reward_terms(state, values)
        weighted = {name: value * self.cfg.reward.scales[name] for name, value in raw.items()}
        reward = np.sum(np.stack(tuple(weighted.values()), axis=0), axis=0).astype(np.float32)

        wheel_omega = values["wheel_omega"]
        wheel_differential = np.mean(wheel_omega[:, 2:], axis=1) - np.mean(
            wheel_omega[:, :2], axis=1
        )
        state.info["Reward"] = weighted
        state.info["metrics"] = {
            "phase": values["phase"].copy(),
            "target_yaw_rate": values["target_yaw_rate"],
            "yaw_rate": values["gyro"][:, 2],
            "yaw_rate_error": np.abs(values["gyro"][:, 2] - values["target_yaw_rate"]),
            "planar_speed": np.linalg.norm(values["lin_vel"][:, :2], axis=1),
            "wheel_differential": wheel_differential,
            "both_feet_grounded": np.all(state.info["contacts"], axis=1).astype(np.float32),
            "spin_angle": state.info["spin_angle"].copy(),
            "last_cycle_angle": state.info["last_cycle_angle"].copy(),
            "root_height": values["root_pose"][:, 2],
            "head_velocity_rms": np.sqrt(
                np.mean(np.square(values["dof_vel"][:, HEAD_SLICE]), axis=1)
            ),
            "head_height": values["head_height"],
            "head_clearance": values["head_height"] - values["root_pose"][:, 2],
            "leg_antisymmetry_scale": np.full(
                self.num_envs,
                self._leg_antisymmetry_scale(),
                dtype=np.float32,
            ),
            "mean_action_abs": np.mean(np.abs(state.info["current_actions"]), axis=1),
        }
        return state.replace(reward=reward)

    def _reward_values(self, state: NpEnvState) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward
        phase = state.info["phase"]
        target_yaw_rate = spin_rate_by_phase(
            phase,
            rate_max=cfg.spin_rate_max,
            accel_end=cfg.accel_end,
            hold_end=cfg.hold_end,
            brake_end=cfg.brake_end,
        )
        return {
            "phase": phase,
            "target_yaw_rate": target_yaw_rate,
            "spin_gate": target_yaw_rate / cfg.spin_rate_max,
            "rest_gate": (phase >= cfg.brake_end).astype(np.float32),
            "dof_pos": self.get_dof_pos(state.data),
            "dof_vel": self.get_dof_vel(state.data),
            "lin_vel": self.get_local_linvel(state.data),
            "gyro": self.get_gyro(state.data),
            "gravity": self._projected_gravity(state.data),
            "root_pose": self._body.get_pose(state.data),
            "head_height": self._head_height_site.get_position(state.data)[:, 2],
            "wheel_omega": self.get_wheel_vel(state.data),
        }

    def _reward_terms(
        self, state: NpEnvState, values: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward
        info = state.info
        values = self._reward_values(state) if values is None else values
        phase = values["phase"]
        target_yaw_rate = values["target_yaw_rate"]
        spin_gate = values["spin_gate"]
        rest_gate = values["rest_gate"]
        dof_pos = values["dof_pos"]
        dof_vel = values["dof_vel"]
        lin_vel = values["lin_vel"]
        gyro = values["gyro"]
        gravity = values["gravity"]
        root_pose = values["root_pose"]
        wheel_omega = values["wheel_omega"]

        tilt = np.sum(np.square(gravity[:, :2]), axis=1)
        pose_error = np.mean(np.square(dof_pos - self.default_angles), axis=1)
        yaw_rate_error = gyro[:, 2] - target_yaw_rate
        drift_scale = np.where(
            phase < cfg.accel_end,
            cfg.launch_drift_scale,
            1.0,
        )
        leg_antisymmetry = -np.mean(
            np.abs(
                dof_pos[:, SCISSOR_LEFT_INDICES]
                - dof_pos[:, SCISSOR_RIGHT_INDICES]
            ),
            axis=1,
        )
        leg_antisymmetry *= spin_gate * self._leg_antisymmetry_scale()
        head_clearance = values["head_height"] - root_pose[:, 2]
        low_head_error = np.maximum(cfg.min_head_clearance - head_clearance, 0.0)
        return {
            "spin_rate_track": np.exp(-np.square(yaw_rate_error / cfg.spin_rate_std)),
            "spin_rate_l1": -np.abs(yaw_rate_error),
            "stay_in_place": np.sum(np.square(lin_vel[:, :2]), axis=1) * drift_scale,
            "wheel_differential": spin_wheel_differential_reward(
                wheel_omega,
                spin_gate,
                cfg.wheel_omega_scale,
            ),
            "leg_antisymmetry": leg_antisymmetry,
            "both_feet_grounded": np.all(info["contacts"], axis=1).astype(np.float32)
            * spin_gate,
            "upright": np.exp(-tilt / cfg.upright_std**2),
            "rest_pose": rest_gate * np.exp(-pose_error / cfg.roller_pose_std**2),
            "base_height": np.exp(
                -np.square(root_pose[:, 2] - cfg.target_root_height) / cfg.root_height_std**2
            ),
            "head_stability": np.exp(
                -np.mean(np.square(dof_vel[:, HEAD_SLICE]), axis=1)
                / cfg.head_velocity_std**2
            ),
            "head_height": np.exp(-np.square(low_head_error) / cfg.head_height_std**2),
            "feet_flat": self._feet_flat_cost(state),
            "vertical_velocity": np.square(lin_vel[:, 2]),
            # Do not penalize yaw: z angular velocity is the task itself.
            "body_ang_vel": np.sum(np.square(gyro[:, :2]), axis=1),
            "action_rate": np.sum(
                np.square(info["current_actions"] - info["last_actions"]), axis=1
            ),
            "joint_velocity": np.sum(np.square(dof_vel), axis=1),
            "joint_acceleration": np.sum(
                np.square((dof_vel - info["last_dof_vel"]) / self.cfg.ctrl_dt), axis=1
            ),
            "joint_limits": self._joint_limit_cost(dof_pos),
            "termination": state.terminated.astype(np.float32),
        }
