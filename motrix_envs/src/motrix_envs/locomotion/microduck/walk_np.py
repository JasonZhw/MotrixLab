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

"""Microduck flat-terrain walking environment for the MotrixSim backend."""

import math

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.microduck.cfg import ENV_NAME, JOINT_NAMES, MicroduckWalkNpEnvCfg
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

NUM_ACTIONS = 14
NUM_OBSERVATIONS = 61
NUM_COMMANDS = 13
HEAD_SLICE = slice(5, 9)
LEG_INDICES = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int64)


def centered_velocity_tracking(
    commands: np.ndarray,
    velocities: np.ndarray,
    tracking_std: float,
) -> np.ndarray:
    """Velocity tracking advantage relative to staying still.

    A conventional Gaussian gives a substantial positive reward at zero
    velocity when the requested speed is small. Subtracting that stationary
    baseline makes standing still worth exactly zero for non-zero commands,
    while retaining a smooth gradient toward the target velocity.
    """
    command_error = np.sum(np.square(commands - velocities), axis=1)
    stationary_error = np.sum(np.square(commands), axis=1)
    tracking = np.exp(-command_error / tracking_std**2)
    stationary = np.exp(-stationary_error / tracking_std**2)
    moving_command = np.linalg.norm(commands, axis=1) > 0.05
    advantage = (tracking - stationary) / np.maximum(1.0 - stationary, 0.05)
    return np.where(moving_command, advantage, tracking).astype(np.float32)


def directional_velocity_reward(commands: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Signed progress along the requested planar velocity direction."""
    command_sq = np.sum(np.square(commands), axis=1)
    progress = np.sum(commands * velocities, axis=1) / np.maximum(command_sq, 2.5e-3)
    progress = np.clip(progress, -1.0, 1.0)
    return np.where(command_sq > 2.5e-3, progress, 0.0).astype(np.float32)


@registry.env(ENV_NAME, sim_backend="np")
class MicroduckWalkTask(NpEnv):
    """Train Microduck to track planar velocity and head-pose commands.

    The actor observation contract intentionally matches the official runtime:
    48 proprioceptive values followed by ``twist(3), head_pose(4), body_pose(6)``.
    """

    def __init__(self, cfg: MicroduckWalkNpEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._body = self._model.get_body(cfg.asset.body_name)

        if tuple(self._model.actuator_names) != JOINT_NAMES:
            raise ValueError(
                "Microduck actuator order changed; the 14D deployment contract would be invalid: "
                f"{self._model.actuator_names}"
            )
        if self._model.num_actuators != NUM_ACTIONS:
            raise ValueError(f"Expected {NUM_ACTIONS} Microduck actuators, got {self._model.num_actuators}")

        # Resolve policy joints by name: roller models interleave four passive
        # wheel joints, but the deployment policy remains a 14-servo contract.
        actuated_joints = tuple(self._model.get_joint(name) for name in JOINT_NAMES)
        self._actuated_qpos_indices = np.asarray(
            [joint.dof_pos_index for joint in actuated_joints], dtype=np.int64
        )
        self._actuated_qvel_indices = np.asarray(
            [joint.dof_vel_index for joint in actuated_joints], dtype=np.int64
        )

        self._action_space = gym.spaces.Box(-1.0, 1.0, (NUM_ACTIONS,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (NUM_OBSERVATIONS,),
            dtype=np.float32,
        )
        self.default_angles = np.asarray(cfg.init_state.default_joint_angles, dtype=np.float32)
        self.joint_limits = np.asarray(
            [
                [joint.range[0, 0] for joint in actuated_joints],
                [joint.range[0, 1] for joint in actuated_joints],
            ],
            dtype=np.float32,
        )
        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self._init_dof_pos = np.asarray(self._model.compute_init_dof_pos(), dtype=np.float32)
        self._init_dof_pos[:3] = (0.0, 0.0, cfg.init_state.root_height)
        self._init_dof_pos[3:7] = (0.0, 0.0, 0.0, 1.0)
        self._init_dof_pos[self._actuated_qpos_indices] = self.default_angles
        self._init_dof_vel = np.zeros(self._model.num_dof_vel, dtype=np.float32)
        self._feet = tuple(self._model.get_site(name) for name in ("left_foot", "right_foot"))
        self._global_step = 0
        self._last_command_stage = 0

        ground = self._model.get_geom_index(cfg.asset.ground_name)
        self._foot_contact_pairs = np.asarray(
            [[self._model.get_geom_index(name), ground] for name in cfg.asset.foot_names],
            dtype=np.uint32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_pos[:, self._actuated_qpos_indices]

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_vel[:, self._actuated_qvel_indices]

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.local_lin_vel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.gyro, data)

    def _projected_gravity(self, data: mtx.SceneData) -> np.ndarray:
        base_quat = self._body.get_pose(data)[:, 3:7]
        return quaternion.rotate_inverse(base_quat, self.gravity_vec)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, NUM_ACTIONS):
            raise ValueError(f"Expected action shape {(self.num_envs, NUM_ACTIONS)}, got {actions.shape}")

        clipped_actions = np.clip(actions, -1.0, 1.0)
        state.info["last_dof_vel"] = self.get_dof_vel(state.data).copy()
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = clipped_actions

        targets = self.default_angles + self.cfg.control.action_scale * clipped_actions
        targets = np.clip(targets, self.joint_limits[0], self.joint_limits[1])
        state.data.actuator_ctrls = targets
        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        state = self.update_observation(state)
        state = self.update_terminated(state)
        return self.update_reward(state)

    def _uniform(self, bounds: tuple[float, float], size: int) -> np.ndarray:
        return np.random.uniform(bounds[0], bounds[1], size=size).astype(np.float32)

    def _sample_commands(self, num_envs: int) -> np.ndarray:
        cfg = self.cfg.commands
        stage = self._command_stage()
        commands = np.empty((num_envs, NUM_COMMANDS), dtype=np.float32)
        if stage == 0:
            # Bootstrap a real gait with unambiguous forward commands. Tiny
            # lateral/yaw ranges keep those input weights alive for stage 1.
            commands[:, 0] = self._uniform((0.15, 0.30), num_envs)
            commands[:, 1] = self._uniform((-0.02, 0.02), num_envs)
            commands[:, 2] = self._uniform((-0.10, 0.10), num_envs)
            zero_probability = 0.05
            head_scale = 0.25
        elif stage == 1:
            commands[:, 0] = self._uniform((-0.10, 0.35), num_envs)
            commands[:, 1] = self._uniform((-0.12, 0.12), num_envs)
            commands[:, 2] = self._uniform((-0.50, 0.50), num_envs)
            zero_probability = 0.15
            head_scale = 0.50
        else:
            commands[:, 0] = self._uniform(cfg.lin_vel_x, num_envs)
            commands[:, 1] = self._uniform(cfg.lin_vel_y, num_envs)
            commands[:, 2] = self._uniform(cfg.ang_vel_z, num_envs)
            zero_probability = cfg.zero_twist_probability
            head_scale = 1.0
        for index, bounds in enumerate(cfg.head_pose, start=3):
            commands[:, index] = self._uniform((bounds[0] * head_scale, bounds[1] * head_scale), num_envs)
        for index, bounds in enumerate(cfg.body_pose, start=7):
            commands[:, index] = self._uniform(bounds, num_envs)

        # Deployment idle is exactly zero, which uniform sampling would never hit.
        zero_twist = np.random.random(num_envs) < zero_probability
        commands[zero_twist, :3] = 0.0
        return commands

    def _command_stage(self) -> int:
        cfg = self.cfg.commands
        if self._global_step < cfg.forward_only_steps:
            return 0
        if self._global_step < cfg.mixed_motion_steps:
            return 1
        return 2

    def set_training_iteration(self, iteration: int, steps_per_iteration: int) -> None:
        """Restore command-curriculum progress when PPO resumes."""
        self._global_step = max(0, iteration * steps_per_iteration)
        self._last_command_stage = self._command_stage()

    def _sample_command_steps(self, num_envs: int) -> np.ndarray:
        low_s, high_s = self.cfg.commands.resampling_time
        low = max(1, int(round(low_s / self.cfg.ctrl_dt)))
        high = max(low + 1, int(round(high_s / self.cfg.ctrl_dt)) + 1)
        return np.random.randint(low, high, size=num_envs, dtype=np.int64)

    def _update_commands(self, info: dict):
        self._global_step += 1
        info["command_steps_remaining"] -= 1
        due = info["command_steps_remaining"] <= 0
        stage = self._command_stage()
        if stage != self._last_command_stage:
            due = np.ones_like(due, dtype=bool)
            self._last_command_stage = stage
        count = int(np.count_nonzero(due))
        if count:
            info["commands"][due] = self._sample_commands(count)
            info["command_steps_remaining"][due] = self._sample_command_steps(count)

    def _observation_noise(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg.noise
        if not cfg.enabled:
            return obs

        num_obs = obs.shape[0]
        noisy = obs.copy()
        noisy[:, 0:3] += np.random.uniform(-cfg.gyro, cfg.gyro, size=(num_obs, 3))
        noisy[:, 3:6] += np.random.uniform(-cfg.gravity, cfg.gravity, size=(num_obs, 3))
        noisy[:, 6:20] += np.random.uniform(-cfg.joint_pos, cfg.joint_pos, size=(num_obs, 14))
        noisy[:, 20:34] += np.random.uniform(-cfg.joint_vel, cfg.joint_vel, size=(num_obs, 14))
        return noisy.astype(np.float32)

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        norm = self.cfg.normalization
        obs = np.hstack(
            (
                self.get_gyro(data) * norm.gyro,
                self._projected_gravity(data) * norm.gravity,
                (self.get_dof_pos(data) - self.default_angles) * norm.joint_pos,
                self.get_dof_vel(data) * norm.joint_vel,
                info["current_actions"] * norm.last_action,
                info["commands"],
            )
        ).astype(np.float32)
        if obs.shape != (data.shape[0], NUM_OBSERVATIONS):
            raise RuntimeError(f"Microduck observation contract is invalid: {obs.shape}")
        return self._observation_noise(obs)

    def _get_foot_contacts(self, data: mtx.SceneData) -> np.ndarray:
        contacts = self._model.get_contact_query(data).is_colliding(self._foot_contact_pairs)
        return contacts.reshape((self.num_envs, 2))

    def _joint_limit_cost(self, dof_pos: np.ndarray) -> np.ndarray:
        joint_range = self.joint_limits[1] - self.joint_limits[0]
        limit_margin = np.maximum(0.1 * joint_range, 1.0e-3)
        limit_distance = np.minimum(dof_pos - self.joint_limits[0], self.joint_limits[1] - dof_pos)
        return np.sum(
            np.square(np.clip((limit_margin - limit_distance) / limit_margin, 0.0, 1.0)),
            axis=1,
        )

    def update_observation(self, state: NpEnvState) -> NpEnvState:
        contacts = self._get_foot_contacts(state.data)
        foot_positions = np.stack([foot.get_position(state.data) for foot in self._feet], axis=1)
        state.info["foot_velocity"] = (foot_positions - state.info["foot_positions"]) / self.cfg.ctrl_dt
        state.info["foot_positions"] = foot_positions

        air_time = state.info["feet_air_time"] + self.cfg.ctrl_dt
        first_contact = np.logical_and(contacts, np.logical_not(state.info["contacts"]))
        state.info["landing_air_time"] = np.where(first_contact, air_time, 0.0)
        state.info["feet_air_time"] = np.where(contacts, 0.0, air_time)
        state.info["swing_accum"] += np.logical_not(contacts) * self.cfg.ctrl_dt
        state.info["contacts"] = contacts
        return state.replace(obs=self._get_obs(state.data, state.info))

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        pose = self._body.get_pose(state.data)
        projected_gravity = self._projected_gravity(state.data)
        max_tilt_cos = math.cos(math.radians(self.cfg.termination.max_tilt_deg))
        finite = np.logical_and(
            np.all(np.isfinite(state.data.dof_pos), axis=1),
            np.all(np.isfinite(state.data.dof_vel), axis=1),
        )
        finite = np.logical_and(finite, np.all(np.isfinite(state.obs), axis=1))
        too_low = pose[:, 2] < self.cfg.termination.min_root_height
        too_tilted = projected_gravity[:, 2] > -max_tilt_cos
        return state.replace(terminated=np.logical_or.reduce((np.logical_not(finite), too_low, too_tilted)))

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        init = self.cfg.init_state
        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        dof_pos[:, :2] += np.random.uniform(-init.root_xy_noise, init.root_xy_noise, size=(num_reset, 2))
        roll = np.random.uniform(-math.radians(init.max_roll_deg), math.radians(init.max_roll_deg), num_reset)
        pitch = np.random.uniform(-math.radians(init.max_pitch_deg), math.radians(init.max_pitch_deg), num_reset)
        yaw = np.random.uniform(-math.pi, math.pi, num_reset)
        dof_pos[:, 3:7] = quaternion.from_euler(roll, pitch, yaw)
        dof_pos[:, self._actuated_qpos_indices] += np.random.uniform(
            -init.joint_pos_noise,
            init.joint_pos_noise,
            size=(num_reset, NUM_ACTIONS),
        )
        dof_pos[:, self._actuated_qpos_indices] = np.clip(
            dof_pos[:, self._actuated_qpos_indices],
            self.joint_limits[0],
            self.joint_limits[1],
        )
        dof_vel[:, self._actuated_qvel_indices] = np.random.uniform(
            -init.joint_vel_noise,
            init.joint_vel_noise,
            size=(num_reset, NUM_ACTIONS),
        )

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        foot_positions = np.stack([foot.get_position(data) for foot in self._feet], axis=1)

        info = {
            "current_actions": np.zeros((num_reset, NUM_ACTIONS), dtype=np.float32),
            "last_actions": np.zeros((num_reset, NUM_ACTIONS), dtype=np.float32),
            "last_dof_vel": np.zeros((num_reset, NUM_ACTIONS), dtype=np.float32),
            "commands": self._sample_commands(num_reset),
            "command_steps_remaining": self._sample_command_steps(num_reset),
            "feet_air_time": np.zeros((num_reset, 2), dtype=np.float32),
            "landing_air_time": np.zeros((num_reset, 2), dtype=np.float32),
            "contacts": np.zeros((num_reset, 2), dtype=bool),
            "foot_positions": foot_positions,
            "foot_velocity": np.zeros((num_reset, 2, 3), dtype=np.float32),
            "swing_accum": np.zeros((num_reset, 2), dtype=np.float32),
        }
        return self._get_obs(data, info), info

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        raw = self._reward_terms(state)
        weighted = {name: value * self.cfg.reward.scales[name] for name, value in raw.items()}
        reward = np.sum(np.stack(tuple(weighted.values()), axis=0), axis=0).astype(np.float32)
        state.info["Reward"] = weighted
        state.info["metrics"] = {
            "root_height": self._body.get_pose(state.data)[:, 2],
            "command_speed": np.linalg.norm(state.info["commands"][:, :2], axis=1),
            "forward_velocity": self.get_local_linvel(state.data)[:, 0],
            "linear_velocity_error": np.linalg.norm(
                state.info["commands"][:, :2] - self.get_local_linvel(state.data)[:, :2],
                axis=1,
            ),
            "foot_air_fraction": np.mean(np.logical_not(state.info["contacts"]), axis=1),
            "single_support_fraction": (np.sum(state.info["contacts"], axis=1) == 1).astype(np.float32),
            "mean_action_abs": np.mean(np.abs(state.info["current_actions"]), axis=1),
            "command_stage": np.full(self.num_envs, self._command_stage(), dtype=np.float32),
        }
        return state.replace(reward=reward)

    def _reward_terms(self, state: NpEnvState) -> dict[str, np.ndarray]:
        data = state.data
        info = state.info
        cfg = self.cfg.reward
        commands = info["commands"]
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        gyro = self.get_gyro(data)
        lin_vel = self.get_local_linvel(data)
        gravity = self._projected_gravity(data)
        pose = self._body.get_pose(data)

        ang_error = np.square(commands[:, 2] - gyro[:, 2])
        tilt = np.sum(np.square(gravity[:, :2]), axis=1)
        head_target = self.default_angles[HEAD_SLICE] + commands[:, 3:7]
        head_error = np.square(dof_pos[:, HEAD_SLICE] - head_target)

        moving = np.linalg.norm(commands[:, :3], axis=1) > 0.01
        pose_std = np.where(moving, cfg.walking_pose_std, cfg.standing_pose_std)
        leg_error = np.mean(np.square(dof_pos[:, LEG_INDICES] - self.default_angles[LEG_INDICES]), axis=1)

        landing_time = info["landing_air_time"]
        airtime_window = max(cfg.air_time_max - cfg.air_time_min, 1.0e-6)
        air_time = np.sum(
            np.clip((landing_time - cfg.air_time_min) / airtime_window, 0.0, 1.0),
            axis=1,
        )
        air_time *= moving

        current_air_time = info["feet_air_time"]
        swing_time = np.sum(
            np.logical_and(current_air_time >= cfg.air_time_min, current_air_time <= cfg.air_time_max),
            axis=1,
        ).astype(np.float32)
        swing_time *= moving

        foot_height = np.maximum(info["foot_positions"][:, :, 2], 0.0)
        airborne = np.logical_not(info["contacts"])
        foot_clearance = np.sum(
            np.exp(-np.square(foot_height - cfg.target_foot_height) / cfg.foot_height_std**2) * airborne,
            axis=1,
        )
        foot_clearance *= moving
        single_support = (np.sum(info["contacts"], axis=1) == 1).astype(np.float32) * moving
        double_support = np.all(info["contacts"], axis=1).astype(np.float32) * moving
        swing_sum = np.sum(info["swing_accum"], axis=1)
        gait_balance = np.abs(info["swing_accum"][:, 0] - info["swing_accum"][:, 1]) / (swing_sum + 1.0e-3)
        foot_slip = np.sum(
            np.sum(np.square(info["foot_velocity"][:, :, :2]), axis=2) * info["contacts"],
            axis=1,
        )

        return {
            "track_linear_velocity": centered_velocity_tracking(
                commands[:, :2], lin_vel[:, :2], cfg.linear_tracking_std
            ),
            "directional_velocity": directional_velocity_reward(commands[:, :2], lin_vel[:, :2]),
            "track_angular_velocity": np.exp(-ang_error / cfg.angular_tracking_std**2),
            "head_pose_tracking": np.mean(np.exp(-head_error / cfg.head_tracking_std**2), axis=1),
            "upright": np.exp(-tilt / cfg.upright_std**2),
            "pose": np.exp(-leg_error / np.square(pose_std)),
            "base_height": np.exp(-np.square(pose[:, 2] - cfg.target_root_height) / cfg.root_height_std**2),
            "air_time": air_time,
            "swing_time": swing_time,
            "foot_clearance": foot_clearance,
            "single_support": single_support,
            "double_support": double_support,
            "gait_balance": gait_balance,
            "foot_slip": foot_slip,
            "vertical_velocity": np.square(lin_vel[:, 2]),
            "body_ang_vel": np.sum(np.square(gyro[:, :2]), axis=1),
            "action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "joint_velocity": np.sum(np.square(dof_vel), axis=1),
            "joint_acceleration": np.sum(
                np.square((dof_vel - info["last_dof_vel"]) / self.cfg.ctrl_dt),
                axis=1,
            ),
            "joint_limits": self._joint_limit_cost(dof_pos),
            "termination": state.terminated.astype(np.float32),
        }
