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
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg


@registry.env("anymal_c_navigation_flat", "np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg

    def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        # Get target marker body
        self._target_marker_body = self._model.get_body("target_marker")

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # Observation space: linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) + last_actions(12) +
        # commands(3) + position_error(2) + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )

        self._init_buffer()

    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        # PD parameters controlled by kp and kv in XML

        # Normalization coefficients
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel], dtype=np.float32
        )

        # Set default joint angles
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action :] = self.default_angles

    def _init_contact_geometry(self):
        """Initialize geometry indices required for contact detection"""
        cfg = self._cfg
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)

        # Initialize contact detection matrix
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """Initialize termination contact detection"""
        cfg = self._cfg
        # Find base geometries
        base_indices = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            try:
                base_idx = self._model.get_geom_index(base_name)
                if base_idx is not None:
                    base_indices.append(base_idx)
                else:
                    print(f"Warning: Geom '{base_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding base geom '{base_name}': {e}")

        # Create base-ground contact detection matrix
        if base_indices:
            self.termination_contact = np.array([[idx, self.ground_index] for idx in base_indices], dtype=np.uint32)
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            # Use empty array
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("Warning: No base contacts configured for termination")

    def _init_foot_contact(self):
        """Initialize foot contact detection"""
        cfg = self._cfg
        foot_indices = []
        for foot_name in cfg.asset.foot_names:
            try:
                foot_idx = self._model.get_geom_index(foot_name)
                if foot_idx is not None:
                    foot_indices.append(foot_idx)
                else:
                    print(f"Warning: Foot geom '{foot_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding foot geom '{foot_name}': {e}")

        # Create foot-ground contact detection matrix
        if foot_indices:
            self.foot_contact_check = np.array([[idx, self.ground_index] for idx in foot_indices], dtype=np.uint32)
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0
            print("Warning: No foot contacts configured")

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        """
        Extract root state from self._body
        """
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        # Get velocity from sensor
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # Save current action for incremental control
        if "current_action" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # Position control mode: directly input target angles
        actions_scaled = actions * self._cfg.control_config.action_scale
        state.data.actuator_ctrls = self.default_angles + actions_scaled
        return state

    def update_state(self, state: NpEnvState):
        data = state.data

        # Get root state
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # Joint states (leg joints)
        joint_pos = self.get_dof_pos(data)  # [num_envs, 12]
        joint_vel = self.get_dof_vel(data)  # [num_envs, 12]
        joint_pos_rel = joint_pos - self.default_angles

        # Get sensor data
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # Get commands - convert to relative velocity commands
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # Calculate desired velocity (based on position error)
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)  # Simple P controller
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # Velocity is 0 after reaching

        # Calculate desired angular velocity (based on heading error)
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        reached_all = np.logical_and(reached_position, reached_heading)

        # Angular velocity command calculation + deadband
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)

        # Reset to zero after reaching
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        state.info["desired_vel_xy"] = desired_vel_xy

        # Combine into velocity commands
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        # Normalize observations
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # Calculate task-related observations
        position_error_normalized = position_error / 5.0  # Normalize to reasonable range
        heading_error_normalized = heading_diff / np.pi  # Normalize to [-1, 1]
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)  # Normalize distance
        reached_flag = reached_all.astype(np.float32)  # Whether target is reached

        # Calculate if zero_ang standard is met: reached and angular velocity close to zero
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,  # 12
                last_actions,  # 12
                command_normalized,  # 3
                position_error_normalized,  # 2 - Position error vector to target
                heading_error_normalized[:, np.newaxis],  # 1 - Heading error
                distance_normalized[:, np.newaxis],  # 1 - Distance to target
                reached_flag[:, np.newaxis],  # 1 - Whether reached
                stop_ready_flag[:, np.newaxis],  # 1 - Whether stop standard is met
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)

        # Update target position marker
        self._update_target_marker(data, pose_commands)
        # Update arrow visualization (no physical effect)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        # Calculate reward
        reward = self._compute_reward(data, state.info, velocity_commands)

        # Calculate termination conditions
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state

    def _update_heading_arrows(
        self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray
    ):
        """
        Update arrow positions (using DOF to control freejoint, no physical effect)
        robot_pos: [num_envs, 3] - Robot position
        desired_vel_xy: [num_envs, 2] - Desired linear velocity (ground coordinates)
        base_lin_vel_xy: [num_envs, 2] - Actual linear velocity (ground coordinates)
        """

        arrow_height = 0.76  # Arrow height (base=0.56 + 0.2)
        cur_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )
        robot_arrow_pos = robot_pos.copy()
        robot_arrow_pos[:, 2] = arrow_height
        robot_arrow_quat = quaternion.from_euler(0, 0, cur_yaw)
        mocap = self._model.get_body("robot_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, robot_arrow_quat], axis=1))

        des_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6, np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]), 0.0
        )
        desired_arrow_quat = quaternion.from_euler(0, 0, des_yaw)
        mocap = self._model.get_body("desired_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, desired_arrow_quat], axis=1))

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        Velocity tracking reward mechanism
        velocity_commands: [num_envs, 3] - (vx, vy, vyaw)
        """
        # Calculate termination condition penalties
        termination_penalty = np.zeros(self._num_envs, dtype=np.float32)

        # Check if DOF velocity exceeds limit
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)

        # Robot base contacts ground penalty
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)
        termination_penalty = np.where(base_contact, -20.0, termination_penalty)

        # Side flip penalty
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)

        # 1. Linear velocity tracking reward
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25

        # 2. Angular velocity tracking reward / heading error penalty (mixed strategy)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        # Get robot position and heading for arrival determination
        robot_position = pose[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)
        target_position = info["pose_commands"][:, :2]
        target_heading = info["pose_commands"][:, 2]
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold

        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold
        reached_all = np.logical_and(reached_position, reached_heading)

        # One-time reward for first time reaching position
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)

        # Distance approach reward: incentivize getting closer to target
        # Use historical minimum distance to calculate progress
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # Reward 5 points for every 1 meter closer

        # 3. Orientation stability reward (penalize deviation from normal standing posture)
        # When standing normally, projected_gravity ≈ [0, 0, -1]
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = (
            np.square(projected_gravity[:, 0])
            + np.square(projected_gravity[:, 1])
            + np.square(projected_gravity[:, 2] + 1.0)
        )

        # Arrival and stop determination (reward bonus)
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # Relax to 0.05 rad/s ≈ 2.86°/s
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        stop_base = 2 * (0.8 * np.exp(-((speed_xy / 0.2) ** 2)) + 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4)))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # 4. Z-axis linear velocity penalty
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # 5. XY-axis angular velocity penalty
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 6. Torque penalty
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)

        # 7. Joint velocity penalty
        joint_vel = self.get_dof_vel(data)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 8. Action change penalty
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # Combined reward
        # After reaching: stop all positive rewards, only keep stop reward and penalties
        reward = np.where(
            reached_all,
            # After reaching: only stop reward and penalties
            (
                stop_bonus
                + arrival_bonus
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # Termination condition penalty
            ),
            # Not reached: normal rewards
            (
                1.5 * tracking_lin_vel  # Increase linear velocity tracking weight
                + 0.3 * tracking_ang_vel  # Decrease angular velocity weight
                + approach_reward  # Approach reward
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # Termination condition penalty
            ),
        )

        return reward

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        Update position and orientation of target marker
        """
        num_envs = data.shape[0]
        arrow_pos = pose_commands.copy()
        arrow_pos[:, 2] = 0.05
        arrow_pos = np.column_stack([pose_commands[:, 0], pose_commands[:, 1], np.full((num_envs, 1), 0.5)])
        arrow_quat = quaternion.from_euler(0, 0, pose_commands[:, 2])
        mocap = self._model.get_body("target_marker").mocap
        mocap.set_pose(data, np.concatenate([arrow_pos, arrow_quat], axis=1))

    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        terminated = np.zeros(self._num_envs, dtype=bool)

        # Check if DOF velocity exceeds limit (prevent inf/numerical divergence)
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        # Extreme velocity/NaN/Inf protection
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # Robot base contacts ground termination
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)
        terminated = np.logical_or(terminated, base_contact)

        # Side flip termination: tilt angle exceeds 75°
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        terminated = np.logical_or(terminated, side_flip_mask)

        return state.replace(terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        cfg: AnymalCEnvCfg = self._cfg
        num_envs = data.shape[0]

        # First generate robot initial position (in world coordinates)
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0],
            pos_range[2],  # x_min, x_max
            num_envs,
        ).astype(np.float32)
        robot_init_y = np.random.uniform(
            pos_range[1],
            pos_range[3],  # y_min, y_max
            num_envs,
        ).astype(np.float32)
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # Generate target position: offset relative to robot initial position
        # pose_command_range now represents offset range relative to robot
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2], high=cfg.commands.pose_command_range[3:5], size=(num_envs, 2)
        ).astype(np.float32)
        target_positions = robot_init_pos + target_offset  # Target position in world coordinates

        # Generate target heading (absolute heading, random in horizontal direction)
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2], high=cfg.commands.pose_command_range[5], size=(num_envs, 1)
        ).astype(np.float32)

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # Set initial state - avoid adding noise to quaternion
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # Create noise - do not add noise to quaternion
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)

        # Base position (DOF 0-2): use the generated random initial position
        noise_pos[:, 0] = robot_init_x - cfg.init_state.pos[0]  # Offset from default position
        noise_pos[:, 1] = robot_init_y - cfg.init_state.pos[1]
        # No noise on Z axis, maintain fixed height to avoid falling feeling

        # All velocities set to 0, ensure completely stationary
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel + noise_vel

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # Update target position marker
        self._update_target_marker(data, pose_commands)

        # Get root state
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # Joint states (leg joints)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # Get sensor data
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # Calculate velocity commands (consistent with update_state)
        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # Position threshold: considered reached within 0.1 meters
        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # Velocity is 0 after reaching

        # Actual linear velocity XY
        base_lin_vel_xy = base_lin_vel[:, :2]

        # Update arrow visualization (no physical effect)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # Heading threshold: considered reached within 15 degrees
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)  # Velocity is 0 after reaching
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)  # Velocity is 0 after reaching

        # Ensure desired_yaw_rate is 1D array
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        # Normalize observations (consistent with update_state)
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        # Calculate task-related observations (consistent with update_state)
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        # Calculate if zero_ang standard is met
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,  # 12
                last_actions,  # 12
                command_normalized,  # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54)

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # Initialize minimum distance
        }

        return obs, info

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quaternion.rotate_vector(quat, gravity)
