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

"""
Shadow Hand Cube Reorientation Environment for MotrixSim

This environment implements the classic in-hand cube manipulation task where the
Shadow Hand must reorient a cube to match random target orientations.

"""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math import quaternion, utils
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import ShadowHandReposeEnvCfg


@registry.env("shadow-hand-repose", sim_backend="np")
class ShadowHandReposeEnv(NpEnv):
    """
    Shadow Hand Cube Reorientation Environment

    Observation space: 157 dimensions
        - 24: hand dof positions (unscaled)
        - 24: hand dof velocities (scaled by 0.2)
        - 7: object pose (pos + quat)
        - 3: object linear velocity
        - 3: object angular velocity (scaled by 0.2)
        - 7: goal pose (pos + quat)
        - 4: relative quaternion (object to goal)
        - 65: fingertip states (5 fingertips * 13: pos + quat + vel)
        - 20: previous actions

    Action space: 20 dimensions (normalized [-1, 1] position targets for actuators)
    """

    _cfg: ShadowHandReposeEnvCfg

    def __init__(self, cfg: ShadowHandReposeEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # Get model info
        self._num_hand_dofs = cfg.num_hand_dofs  # 24 total DOFs
        self._num_actuators = cfg.num_actuators  # 20 actuated joints

        # Initialize spaces
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actuators,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(157,), dtype=np.float32)

        # Get actuator control ranges from model
        self._actuator_ctrl_lower = self._model.actuator_ctrl_limits[0, :]
        self._actuator_ctrl_upper = self._model.actuator_ctrl_limits[1, :]

        # Get joint limits for all DOFs (use model's joint_limits directly)
        self._hand_dof_lower_limits = self._model.joint_limits[0, :]
        self._hand_dof_upper_limits = self._model.joint_limits[1, :]

        # Fingertip link indices (use get_link_index for pose/velocity access)
        self._fingertip_link_ids = []
        for name in cfg.fingertip_link_names:
            link_id = self._model.get_link_index(name)
            self._fingertip_link_ids.append(link_id)
        self._num_fingertips = len(self._fingertip_link_ids)

        # Get cube and target link indices
        self._cube_link_id = self._model.get_link_index("cube")
        self._cube_body = self._model.get_body("cube")
        self._cube_dof_vel_indices = self._cube_body.get_dof_vel_indices()

        # Target is a mocap body - access via Body object (no get_mocap_index API)
        self._target_body = self._model.get_body("target")
        self._target_link_id = self._model.get_link_index("target")
        assert self._target_body.is_mocap, "Target must be a mocap body"

        # Initial cube position (in hand)
        self._in_hand_pos = np.array(cfg.cube_initial_pos, dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _extract_cube_states(self, data: mtx.SceneData, body: mtx.Body):
        return body.get_position(data), body.get_rotation(data), data.dof_vel[:, body.get_dof_vel_indices()]

    def _extract_link_states(self, data: mtx.SceneData, link_ids):
        """
        Extract position, quaternion, and velocity for specified links.

        Args:
            data: SceneData object
            link_ids: List of link indices or single int

        Returns:
            Tuple of (positions, quaternions, velocities)
            - positions: (batch, num_links, 3)
            - quaternions: (batch, num_links, 4) in (x, y, z, w) format
            - velocities: (batch, num_links, 6) [linear_vel, angular_vel]
        """
        # Ensure link_ids is a list
        if isinstance(link_ids, int):
            link_ids = [link_ids]

        # Get all link poses: shape (batch, num_links_total, 7) [x, y, z, qx, qy, qz, qw]
        all_poses = self._model.get_link_poses(data)

        # Extract poses for requested links
        poses = all_poses[:, link_ids, :]  # (batch, num_requested_links, 7)

        # Split into position and quaternion
        positions = poses[:, :, :3]
        quaternions = poses[:, :, 3:]  # (qx, qy, qz, qw)

        # TODO: Velocity computation - MotrixSim doesn't expose link velocities yet
        # Temporary solution: use zero velocities
        # Future options:
        # 1. Finite differences (requires storing previous poses)
        # 2. Compute from DOF velocities via Jacobian (if available)
        # 3. Wait for API: model.get_link_velocities(data)
        batch_size = data.shape[0]
        num_links = len(link_ids)
        velocities = np.zeros((batch_size, num_links, 6), dtype=np.float32)
        for j in np.arange(num_links):
            link = self._model.get_link(link_ids[j])
            velocities[:, j] = np.concatenate(
                (link.get_linear_velocity(data), link.get_angular_velocity(data)), axis=-1
            )

        return positions, quaternions, velocities

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """Apply actions to the hand actuators."""
        cfg = self._cfg

        # Scale actions from [-1, 1] to actuator control range
        targets = utils.scale(actions, self._actuator_ctrl_lower, self._actuator_ctrl_upper)

        # Apply action moving average for smoothness
        if cfg.act_moving_average < 1.0:
            targets = cfg.act_moving_average * targets + (1.0 - cfg.act_moving_average) * state.info["prev_actions"]

        # Clamp to control limits
        targets = np.clip(targets, self._actuator_ctrl_lower, self._actuator_ctrl_upper)

        # Set actuator controls
        state.data.actuator_ctrls = targets
        state.info["prev_actions"] = targets.copy()

        return state

    def update_state(self, state: NpEnvState):
        """Update observations, rewards, and termination conditions."""
        data = state.data
        info = state.info

        # compute obs
        obs = self._compute_observation(state.data, info)

        # Compute reward and termination
        reward, terminated, goal_reached = self._compute_reward(state, info)
        if np.any(goal_reached):
            reset_goal_indices = np.where(goal_reached)[0]
            self._reset_goal_pose(info, reset_goal_indices)
        # Update target visualization
        self._update_target_visualization(data, info)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state

    def _compute_observation(self, data: mtx.SceneData, info: dict):
        cfg = self._cfg
        num_envs = data.shape[0]

        # Get hand DOF states
        hand_dof_pos = data.dof_pos[:, : self._num_hand_dofs]
        hand_dof_vel = data.dof_vel[:, : self._num_hand_dofs]

        # Get cube state using link poses
        cube_pos, cube_quat, cube_vel = self._extract_cube_states(data, self._cube_body)
        cube_linvel = cube_vel[:, :3]
        cube_angvel = cube_vel[:, 3:]

        # Get fingertip states using link poses
        fingertip_pos, fingertip_quat, fingertip_vel = self._extract_link_states(data, self._fingertip_link_ids)

        # Flatten fingertip states (5 × 13 = 65)
        fingertip_state = np.concatenate(
            [
                fingertip_pos.reshape(num_envs, -1),  # 15
                fingertip_quat.reshape(num_envs, -1),  # 20
                fingertip_vel.reshape(num_envs, -1),  # 30
            ],
            axis=-1,
        )  # Total: 65

        # Compute relative quaternion
        relative_quat = quaternion.mul(cube_quat, quaternion.conjugate(info["goal_rot"]))
        scaled_hand_pos = utils.unscale(hand_dof_pos, self._hand_dof_lower_limits, self._hand_dof_upper_limits)

        # Build observation (157 dims)
        return np.concatenate(
            [
                scaled_hand_pos,
                cfg.vel_obs_scale * hand_dof_vel,  # 24
                cube_pos,  # 3
                cube_quat,  # 4
                cube_linvel,  # 3
                cfg.vel_obs_scale * cube_angvel,  # 3
                info["goal_pos"],  # 3
                info["goal_rot"],  # 4
                relative_quat,  # 4
                fingertip_state,  # 65
                info["prev_actions"],  # 20
            ],
            axis=-1,
        )

    def _compute_reward(self, state: NpEnvState, info: dict):
        """
        Reward components (3 core items):
        1. Position distance penalty
        2. Rotation alignment reward
        3. Action regularization penalty

        Additional rewards/penalties:
        - Success bonus when goal is reached
        - Fall penalty when cube drops
        - Timeout penalty when episode ends without success
        """
        cfg = self._cfg
        num_envs = self._num_envs

        # Get cube state using link poses
        cube_pos, cube_quat, _ = self._extract_cube_states(state.data, self._cube_body)

        # Distance from cube to goal position
        goal_dist = np.linalg.norm(cube_pos - state.info["goal_pos"], axis=-1)

        # Rotation distance
        rot_dist = quaternion.rotation_distance(cube_quat, state.info["goal_rot"])

        # Core reward components
        dist_rew = goal_dist * cfg.dist_reward_scale
        rot_rew = 1.0 / (np.abs(rot_dist) + cfg.rot_eps) * cfg.rot_reward_scale
        action_penalty = np.sum(state.info["prev_actions"] ** 2, axis=-1) * cfg.action_penalty_scale

        # Base reward
        reward = dist_rew + rot_rew + action_penalty

        # Check for success (only rotation tolerance)
        goal_reached = np.abs(rot_dist) <= cfg.success_tolerance

        # Update success counter
        info["successes"] += goal_reached * 1

        # Success bonus
        reward = np.where(goal_reached, reward + cfg.reach_goal_bonus, reward)

        # Fall penalty
        fallen = goal_dist >= cfg.fall_dist
        reward = np.where(fallen, reward + cfg.fall_penalty, reward)

        # Termination conditions
        terminated = np.zeros(num_envs, dtype=bool)

        # 1. Fall termination
        terminated = np.logical_or(terminated, fallen)

        # 2. Success termination with hold mechanism
        if cfg.max_consecutive_successes > 0:
            # Reset progress on goal reached when max consecutive successes reached
            new_pos = info["successes"] >= cfg.max_consecutive_successes
            info["successes"] *= 1 - new_pos

        # 3. NaN protection
        terminated = np.logical_or(terminated, np.isnan(rot_dist))
        terminated = np.logical_or(terminated, np.isnan(goal_dist))

        return reward, terminated, new_pos

    def _update_target_visualization(self, data: mtx.SceneData, info: dict):
        """Update the target mocap body to visualize the goal pose."""
        cfg = self._cfg

        # Compute visualization position (offset from goal position)
        viz_pos = info["goal_pos"] + np.array(cfg.viz_target_offset, dtype=np.float32)

        # Combine into pose array: [x, y, z, qx, qy, qz, qw]
        viz_pose = np.concatenate([viz_pos, info["goal_rot"]], axis=-1)

        # Update mocap body pose using correct API
        self._target_body.mocap.set_pose(data, viz_pose)

    def reset(self, data: mtx.SceneData):
        """Reset environments."""
        cfg = self._cfg

        # data is already filtered to contain only envs that need reset
        num_resets = data.shape[0]

        # Reset scene data
        data.reset(self._model)

        # Reset hand DOFs with noise
        init_dof_pos = self._model.compute_init_dof_pos()
        init_dof_vel = np.zeros(self._model.num_dof_vel, dtype=np.float32)

        # Add noise to DOF positions
        dof_pos_noise = np.random.uniform(
            -cfg.reset_dof_pos_noise,
            cfg.reset_dof_pos_noise,
            (num_resets, self._num_hand_dofs),
        ).astype(np.float32)

        # Add noise to DOF velocities
        dof_vel_noise = np.random.uniform(
            -cfg.reset_dof_vel_noise, cfg.reset_dof_vel_noise, (num_resets, self._num_hand_dofs)
        ).astype(np.float32)

        # Set DOF states for all envs in data (already filtered)
        dof_pos = np.tile(init_dof_pos, (num_resets, 1))
        dof_vel = np.tile(init_dof_vel, (num_resets, 1))
        dof_pos[:, : self._num_hand_dofs] += dof_pos_noise
        dof_vel[:, : self._num_hand_dofs] += dof_vel_noise

        data.set_dof_pos(dof_pos, self._model)
        data.set_dof_vel(dof_vel)

        # Reset cube position with small noise
        cube_pos_noise = np.random.uniform(-cfg.reset_position_noise, cfg.reset_position_noise, (num_resets, 3)).astype(
            np.float32
        )
        cube_pos = np.tile(self._in_hand_pos, (num_resets, 1))
        cube_pos += cube_pos_noise

        # Randomize cube orientation
        cube_quat = quaternion.generate_random_shoemake(num_resets)

        # Set cube pose using body's set_dof_pos method
        # Combine into DOF pose: [x, y, z, qx, qy, qz, qw]
        cube_dof_pos = np.concatenate([cube_pos, cube_quat], axis=-1)

        # Set cube DOF position
        self._cube_body.set_dof_pos(data, cube_dof_pos)

        # Set cube DOF velocity to zero
        cube_dof_vel = np.zeros((num_resets, 6), dtype=np.float32)  # 6DOF velocity
        self._cube_body.set_dof_vel(data, cube_dof_vel)

        # Reset goal pose
        # Note: goal_pos and goal_rot are indexed by original env indices
        info = {
            "goal_pos": np.tile(self._in_hand_pos, num_resets).reshape(num_resets, 3),
            "goal_rot": quaternion.generate_random_shoemake(num_resets),
            "prev_actions": np.zeros((num_resets, self._num_actuators), dtype=np.float32),
            "successes": np.zeros((num_resets), dtype=np.int32),
        }

        obs = self._compute_observation(data, info)

        return obs, info

    def _reset_goal_pose(self, info, env_ids):
        """Reset goal pose to random orientation with fixed position."""
        num_resets = len(env_ids)

        # Goal position is fixed

        # Randomize goal orientation using Shoemake method for uniform SO(3) sampling
        info["goal_rot"][env_ids] = quaternion.generate_random_shoemake(num_resets)
