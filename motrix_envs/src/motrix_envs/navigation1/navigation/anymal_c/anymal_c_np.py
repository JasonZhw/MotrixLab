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
import os

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg

@registry.env("anymal_c_navigation_flat","np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg

    def __init__(self, cfg:AnymalCEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs = num_envs)
    
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception as e:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
    
        self._action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (12,), dtype = np.float32)
        # 观测空间：linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) + last_actions(12) + commands(3) + position_error(2) + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (54,), dtype = np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )
        
        # 查找target_marker的DOF索引并更新_init_dof_pos
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引（如果箭头body存在）
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        self._init_buffer()
    
    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype = np.float32)
        # PD参数由XML中的kp和kv控制
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action:] = self.default_angles
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        # 新的DOF结构：
        # DOF 0-2: target_marker (slide x, slide y, hinge yaw)
        # DOF 3-5: base position (x, y, z)
        # DOF 6-9: base quaternion (qx, qy, qz, qw) - Motrix引擎格式
        # DOF 10+: joint angles (12个关节)
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        
        # 设置target_marker的初始位置为原点，朝向为0
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]  # [x, y, yaw]
        
        # base的四元数索引
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        # DOF结构：
        # DOF 0-2: target_marker (3个: slide x, slide y, hinge yaw)
        # DOF 3-5: base position (3个)
        # DOF 6-9: base quaternion (4个)
        # DOF 10-21: joint angles (12个)
        # DOF 22-28: robot_heading_arrow freejoint (7个: 3 pos + 4 quat)
        # DOF 29-35: desired_heading_arrow freejoint (7个: 3 pos + 4 quat)
        
        # robot_heading_arrow的DOF索引
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        
        # desired_heading_arrow的DOF索引
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        # 设置箭头的初始位置和姿态: [x, y, z, qx, qy, qz, qw]
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]
            
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]

    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        cfg = self._cfg
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        
        # 初始化接触检测矩阵
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """初始化终止接触检测"""
        cfg = self._cfg
        # 查找基座几何体
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

        # 创建基座-地面接触检测矩阵
        if base_indices:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            # 使用空数组
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("Warning: No base contacts configured for termination")

    def _init_foot_contact(self):
        """初始化足部接触检测"""
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
        
        # 创建足部-地面接触检测矩阵
        if foot_indices:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices],
                dtype=np.uint32
            )
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
        从self._body中提取根节点状态
        """
        pose = self._body.get_pose(data)
        # 位置 [x, y, z]
        root_pos = pose[:, :3]
        # 四元数 [qx, qy, qz, qw] - Motrix引擎格式
        root_quat = pose[:, 3:7]
        # 使用传感器获取速度
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state
        
    def _compute_torques(self, actions, data):
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        return target_pos

    def update_state(self, state:NpEnvState):
        data = state.data

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态（腿部关节）
        joint_pos = self.get_dof_pos(data)        # [num_envs, 12]
        joint_vel = self.get_dof_vel(data)        # [num_envs, 12]
        joint_pos_rel = joint_pos - self.default_angles

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 获取命令 - 转换为相对速度命令
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算期望速度（基于位置误差）
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)  # 简单P控制器
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 计算期望角速度（基于朝向误差）
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        heading_threshold = np.deg2rad(15) 
        reached_heading = np.abs(heading_diff) < heading_threshold

        reached_all = np.logical_and(reached_position, reached_heading)

        # 角速度命令计算 + 死区
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)

        # 到达后归零
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        

        
        # 组合为速度命令
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        # 计算任务相关观测
        position_error_normalized = position_error / 5.0  # 归一化到合理范围
        heading_error_normalized = heading_diff / np.pi  # 归一化到[-1, 1]
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)  # 归一化距离
        reached_flag = reached_all.astype(np.float32)  # 是否到达目标
        
        # 计算是否达到zero_ang标准：到达且角速度接近零
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2 - 到目标的位置误差向量
                heading_error_normalized[:, np.newaxis],  # 1 - 朝向误差
                distance_normalized[:, np.newaxis],  # 1 - 到目标的距离
                reached_flag[:, np.newaxis],  # 1 - 是否已到达
                stop_ready_flag[:, np.newaxis],  # 1 - 是否达到停止标准
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        # 更新箭头可视化（不影响物理）
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 计算奖励
        reward = self._compute_reward(data, state.info, velocity_commands)

        # 计算终止条件
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        # 调试打印（每200步一次）
        state.info["steps"] = state.info.get("steps", np.zeros(self._num_envs, dtype=np.int32)) + 1
        if state.info["steps"][0] % 200 == 0:
            robot_position = root_pos[:, :2]
            robot_heading = self._get_heading_from_quat(root_quat)
            target_position = pose_commands[:, :2]
            target_heading = pose_commands[:, 2]
            position_error = np.linalg.norm(target_position - robot_position, axis=1)
            heading_diff = target_heading - robot_heading
            heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
            heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
            mean_pos_err = np.mean(position_error)
            mean_heading_err = np.rad2deg(np.mean(np.abs(heading_diff)))
            mean_vel = np.mean(np.linalg.norm(base_lin_vel[:, :2], axis=1))

        
        return state

    def _get_heading_from_quat(self, quat:np.ndarray) -> np.ndarray:
        # Motrix引擎格式: [qx, qy, qz, qw]
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # 计算yaw角（绕Z轴旋转）
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """
        更新箭头位置（使用DOF控制freejoint，不影响物理）
        robot_pos: [num_envs, 3] - 机器人位置
        desired_vel_xy: [num_envs, 2] - 期望线速度（地面坐标）
        base_lin_vel_xy: [num_envs, 2] - 实际线速度（地面坐标）
        """
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_height = 0.76  # 箭头高度（base=0.56 + 0.2）
        
        # 获取所有环境的dof_pos
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 当前运动方向箭头（绿色）：由实际线速度方向决定
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([
                robot_pos[env_idx, 0],
                robot_pos[env_idx, 1],
                arrow_height
            ], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头（蓝色）：由期望线速度方向决定
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([
                robot_pos[env_idx, 0],
                robot_pos[env_idx, 1],
                arrow_height
            ], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _quat_multiply(self, q1, q2):
        """Motrix格式四元数乘法 [qx, qy, qz, qw]"""
        qx1, qy1, qz1, qw1 = q1[0], q1[1], q1[2], q1[3]
        qx2, qy2, qz2, qw2 = q2[0], q2[1], q2[2], q2[3]
        
        qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2
        qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
        qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
        qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """
        欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        速度跟踪奖励机制
        velocity_commands: [num_envs, 3] - (vx, vy, vyaw)
        """
        # 计算终止条件惩罚
        termination_penalty = np.zeros(self._num_envs, dtype=np.float32)
        
        # 检查DOF速度是否超限
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)
        
        # 机器人基座接触地面惩罚
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)
        termination_penalty = np.where(base_contact, -20.0, termination_penalty)
        
        # 侧翻惩罚
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)
        
        # 线速度跟踪奖励
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25
        
        # 角速度跟踪奖励 / 朝向偏差惩罚（混合策略）
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)
        
        # 获取机器人位置和朝向用于到达判定
        robot_position = pose[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = info["pose_commands"][:, :2]
        target_heading = info["pose_commands"][:, 2]
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold
        
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold
        reached_all = np.logical_and(reached_position, reached_heading)
        
        # 首次到达位置的一次性奖励
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)
        
        # 距离接近奖励：激励靠近目标
        # 使用历史最近距离来计算进步
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # 每接近1米奖励5分
        
        # 姿态稳定性奖励（惩罚偏离正常站立姿态）
        # 正常站立时 projected_gravity ≈ [0, 0, -1]
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1]) + np.square(projected_gravity[:, 2] + 1.0)

        # 到达与停止判定（奖励加成）
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # 放宽到0.05 rad/s ≈ 2.86°/s
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        stop_base = 2 * (0.8 * np.exp(- (speed_xy / 0.2)**2) + 1.2 * np.exp(- (np.abs(gyro[:, 2]) / 0.1)**4))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)
        
        # Z轴线速度惩罚
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])
        
        # XY轴角速度惩罚
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)
        
        # 力矩惩罚
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)
        
        # 关节速度惩罚
        joint_vel = self.get_dof_vel(data)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)
        
        # 动作变化惩罚
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)
        
        # 综合奖励
        # 到达后：停止所有正向奖励，只保留停止奖励和惩罚项
        reward = np.where(
            reached_all,
            # 到达后：只有停止奖励和惩罚
            (
                stop_bonus
                + arrival_bonus
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # 终止条件惩罚
            ),
            # 未到达：正常奖励
            (
                1.5 * tracking_lin_vel    # 提高线速度跟踪权重
                + 0.3 * tracking_ang_vel  # 降低角速度权重
                + approach_reward         # 接近奖励
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # 终止条件惩罚
            )
        )
        
        # 调试打印：到达一次性奖励、停止奖励、零角奖励、角速度
        try:
            arrival_count = int((arrival_bonus > 0).sum())
            stop_count = int((stop_bonus > 0).sum())
            zero_ang_count = int((zero_ang_bonus > 0).sum())
            gyro_z_mean = float(np.mean(abs(gyro[:, 2])))
            total_envs = self._num_envs
            
            # 额外统计：环境状态分布
            reached_pos_count = int(reached_position.sum())
            reached_head_count = int(reached_heading.sum())
            dist_mean = float(np.mean(distance_to_target))
            heading_err_mean = float(np.rad2deg(np.mean(np.abs(heading_diff))))
            
            # print(f"[reward_debug] arrival={arrival_count}/{total_envs} stop={stop_count}/{total_envs} zero_ang={zero_ang_count}/{total_envs}")
            # print(f"[position] reached_pos={reached_pos_count}/{total_envs} dist_mean={dist_mean:.2f}m")
            # print(f"[heading] reached_head={reached_head_count}/{total_envs} heading_err_mean={heading_err_mean:.1f}°")
            # print(f"[velocity] gyro_z_mean={gyro_z_mean:.4f} rad/s")
        except Exception:
            pass
        return reward
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        更新目标位置标记的位置和朝向
        pose_commands: [num_envs, 3] - (target_x, target_y, target_heading)
        """
        num_envs = data.shape[0]
        
        # 获取所有环境的dof_pos
        all_dof_pos = data.dof_pos.copy()  # [num_envs, num_dof]
        
        # 为每个环境更新目标标记位置
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])  # 已经是角度，不需要转换
            
            # 更新target_marker的DOF: [x, y, yaw]
            # 只需要设置水平位置和绕Z轴的朝向
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        # 必须调用forward_kinematic才能更新body的pose
        self._model.forward_kinematic(data)

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        # Motrix引擎格式: [qx, qy, qz, qw]
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # 重力向量
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        vx, vy, vz = gravity_world[0], gravity_world[1], gravity_world[2]

        # 计算旋转后向量（四元数旋转公式）
        rx = (1 - 2*(qy*qy + qz*qz)) * vx + 2*(qx*qy - qw*qz) * vy + 2*(qx*qz + qw*qy) * vz
        ry = 2*(qx*qy + qw*qz) * vx + (1 - 2*(qx*qx + qz*qz)) * vy + 2*(qy*qz - qw*qx) * vz
        rz = 2*(qx*qz - qw*qy) * vx + 2*(qy*qz + qw*qx) * vy + (1 - 2*(qx*qx + qy*qy)) * vz
    
        projected_gravity = np.stack([rx, ry, rz], axis = -1)
        return projected_gravity


    def _compute_terminated(self, state:NpEnvState) -> NpEnvState:
        data = state.data
        terminated = np.zeros(self._num_envs, dtype = bool)

        # 超时终止
        timeout = np.zeros(self._num_envs, dtype=bool)
        if self._cfg.max_episode_steps:
            timeout = state.info["steps"] >= self._cfg.max_episode_steps
            terminated = np.logical_or(terminated, timeout)

        # 检查DOF速度是否超限（防止inf/数值发散）
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        # 极端速度/NaN/Inf 保护
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # 机器人基座接触地面终止
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)
        terminated = np.logical_or(terminated, base_contact)
        
        # 侧翻终止：倾斜角度超过75°
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        terminated = np.logical_or(terminated, side_flip_mask)
        
        # 调试：统计终止原因
        if terminated.any():
            timeout_count = int(timeout.sum())
            vel_count = int((vel_overflow | vel_extreme).sum())
            contact_count = int(base_contact.sum())
            flip_count = int(side_flip_mask.sum())
            total = int(terminated.sum())
            if total > 0 and state.info["steps"][0] % 100 == 0:  # 每100步打印一次
                print(f"[termination] total={total} timeout={timeout_count} vel={vel_count} contact={contact_count} flip={flip_count}")
        
        return state.replace(terminated = terminated)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: AnymalCEnvCfg = self._cfg
        num_envs = data.shape[0]

        # 先生成机器人的初始位置（在世界坐标系中）
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0], pos_range[2],  # x_min, x_max
            num_envs
        )
        robot_init_y = np.random.uniform(
            pos_range[1], pos_range[3],  # y_min, y_max
            num_envs
        )
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # 生成目标位置：相对于机器人初始位置的偏移
        # pose_command_range 现在表示相对机器人的偏移范围
        target_offset = np.random.uniform(
            low = cfg.commands.pose_command_range[:2],
            high = cfg.commands.pose_command_range[3:5],
            size = (num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset  # 世界坐标系中的目标位置

        # 生成目标朝向（绝对朝向，水平方向随机）
        target_headings = np.random.uniform(
            low = cfg.commands.pose_command_range[2],
            high = cfg.commands.pose_command_range[5],
            size = (num_envs, 1)
        )

        pose_commands = np.concatenate([target_positions, target_headings],axis = 1)

        # 设置初始状态 - 避免给四元数添加噪声
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # 创建噪声 - 不要给四元数添加噪声
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)
        
        # target_marker (DOF 0-2): 不添加噪声，会在_update_target_marker中设置
        
        # base的位置 (DOF 3-5): 使用前面生成的随机初始位置
        noise_pos[:, 3] = robot_init_x - cfg.init_state.pos[0]  # 相对默认位置的偏移
        noise_pos[:, 4] = robot_init_y - cfg.init_state.pos[1]
        # Z轴不添加噪声，保持固定高度避免坠落感
        # base的四元数 (DOF 6-9): 不添加噪声，保持为单位四元数
        
        # 关节角度(DOF 10:)不添加噪声，保证初始站立稳定
        # noise_pos[:, 10:] = 0.0  # 已经初始化为0

        # 所有速度都设为0，确保完全静止
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel + noise_vel
        
        # 归一化base的四元数（DOF 6-9）
        # 新的DOF结构：target_marker占0-2, base_pos占3-5, base_quat占6-9
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]  # [qx, qy, qz, qw]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:  # 避免除以零
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 默认单位四元数
            
            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                # robot_heading_arrow的四元数（DOF 25-28: qx, qy, qz, qw）
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                # desired_heading_arrow的四元数（DOF 32-35: qx, qy, qz, qw）
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态（腿部关节）
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 计算速度命令（与update_state一致）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)  # [num_envs]

        # 位置阈值：0.1米内认为到达
        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold  # [num_envs]
        
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 实际线速度 XY
        base_lin_vel_xy = base_lin_vel[:, :2]

        # 更新箭头可视化（不影响物理）
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 朝向阈值：15度内认为到达
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold  # [num_envs]
        
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)  # 到达后觗速度为0
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0
        
        # 确保 desired_yaw_rate 是1维数组
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测（与update_state一致）
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        
        # 计算任务相关观测（与update_state一致）
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        # 计算是否达到zero_ang标准
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
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
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 初始化最小距离
        }
        
        return obs, info



