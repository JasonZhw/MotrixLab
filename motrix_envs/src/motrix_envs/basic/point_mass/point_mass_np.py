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

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import PointMassEnvCfg


@registry.env("point_mass", "np")
class PointMassEnv(NpEnv):
    _cfg: PointMassEnvCfg

    def __init__(self, cfg: PointMassEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        self._point_mass = self._model.get_body("point_mass")
        self._target = self._model.get_body("target")

        self._target_radius = cfg.target_radius

        # Target stay counter, used to control reset after 0.5 seconds of overlap
        self._in_target_steps = np.zeros(self._num_envs, dtype=np.int32)
        self._required_in_target_steps = int(0.5 / cfg.ctrl_dt)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.clip(actions, -1.0, 1.0)
        state.data.actuator_ctrls = actions
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        dof_pos = data.dof_pos[:, :2]  # Only x and y positions
        dof_vel = data.dof_vel[:, :2]  # Only x and y velocities

        # Get target position
        target_pos = self._target.get_pose(data)[:, :2]

        # Calculate distance and direction to target
        delta = target_pos - dof_pos
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)

        obs = np.concatenate([dof_pos, dof_vel, target_pos, delta, distance], axis=-1)
        return obs

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)

        # Get positions of point mass and target
        point_pos = self._point_mass.get_pose(data)[:, :2]
        target_pos = self._target.get_pose(data)[:, :2]
        dist_to_target = np.linalg.norm(point_pos - target_pos, axis=-1)

        # Calculate effective target radius for complete overlap
        # Blue ball radius is 0.05, red ball radius is 0.1
        # For complete overlap, center distance should be very small
        effective_target_radius = 0.02  # Smaller radius for complete overlap

        # Fine-grained distance reward - exponential function, reward grows faster as distance decreases
        distance_reward = np.exp(-10 * dist_to_target)  # Stronger exponential reward

        # Large bonus for complete target entry
        in_target = dist_to_target < effective_target_radius
        target_bonus = 100.0 * in_target  # Significantly increased reward

        # Continuous stay reward
        continuous_reward = 30.0 * in_target  # Increased continuous reward

        # Penalty for distance from target center - encourages complete overlap
        # When inside target, penalty increases with distance from center
        center_penalty = np.where(in_target, 10.0 * dist_to_target, 0.0)

        # Control penalty - increased penalty to encourage smoother movement
        dof_vel = data.dof_vel[:, :2]
        vel_magnitude = np.linalg.norm(dof_vel, axis=-1)
        control_penalty = 0.1 * vel_magnitude  # Increased penalty to reduce excessive movement

        # Path optimization reward - encourages straight-line movement
        # Calculate alignment between velocity direction and target direction
        if dist_to_target.max() > 0:
            delta = target_pos - point_pos
            delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
            delta_normalized = delta / delta_norm
            vel_normalized = dof_vel / (np.linalg.norm(dof_vel, axis=-1, keepdims=True) + 1e-6)
            direction_alignment = np.sum(delta_normalized * vel_normalized, axis=-1)
            path_reward = 0.5 * direction_alignment
        else:
            path_reward = 0.0

        # Total reward
        rwd = distance_reward + target_bonus + continuous_reward + path_reward - center_penalty - control_penalty

        # Update target stay steps
        self._in_target_steps = np.where(in_target, self._in_target_steps + 1, 0)

        # Check if stayed in target long enough
        in_target_long_enough = self._in_target_steps >= self._required_in_target_steps

        # Check termination conditions - terminate when reaching target for 0.5 seconds or when NaN encountered
        terminated = np.zeros((self._num_envs,), dtype=bool)
        terminated = np.logical_or(in_target_long_enough, terminated)
        terminated = np.logical_or(np.isnan(obs).any(axis=-1), terminated)

        state.obs = obs
        state.reward = rwd
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num_reset = data.shape[0]

        # Get the actual number of DOF from the model
        num_dof_pos = self._model.num_dof_pos
        num_dof_vel = self._model.num_dof_vel

        # Random initial position within a range for the point mass (only x, y)
        x_pos = np.random.uniform(-1.0, 1.0, size=num_reset).astype(np.float32)
        y_pos = np.random.uniform(-1.0, 1.0, size=num_reset).astype(np.float32)

        # Create dof_pos with the correct length (point mass x, y and target x, y)
        dof_pos = np.zeros((num_reset, num_dof_pos), dtype=np.float32)
        dof_pos[:, 0] = x_pos  # point_mass_x
        dof_pos[:, 1] = y_pos  # point_mass_y

        dof_vel = np.zeros((num_reset, num_dof_vel), dtype=np.float32)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)

        # Randomize target position using its slide joints
        target_x = np.random.uniform(-1.5, 1.5, size=num_reset).astype(np.float32)
        target_y = np.random.uniform(-1.5, 1.5, size=num_reset).astype(np.float32)

        # Set target position via its slide joints (indices 2 and 3)
        dof_pos[:, 2] = target_x  # target_x
        dof_pos[:, 3] = target_y  # target_y
        data.set_dof_pos(dof_pos, self._model)

        self._model.forward_kinematic(data)

        # Reset target stay counter for the environments being reset
        self._in_target_steps[:num_reset] = 0

        obs = self._get_obs(data)
        return obs, {}
