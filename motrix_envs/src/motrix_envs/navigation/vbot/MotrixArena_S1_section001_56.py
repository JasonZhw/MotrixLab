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
VBot 平地导航环境 - MotrixArena_S1_section001_56
====================================================

任务目标：
1. 从外围环形出生区移动到中心目标点
2. 到达后尽快稳定停住
3. 全程避免摔倒与数值发散

奖励设计：Phase-Based（阶段切换）
- Phase 1（未到达）：鼓励“朝目标移动 + 朝向对齐 + 速度跟踪”
- Phase 2（已到达）：鼓励“低速稳定站立 + 抑制抖动 + 防倾斜”

数值稳定策略：
- 统一进行四元数有效性/归一化处理
- 对 NaN / Inf 做防护替换
- 保留发散与越界终止条件
"""

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection001EnvCfg

@registry.env("MotrixArena_S1_section001_56", "np")
class VBotSection001Env(NpEnv):
    """Section001 导航环境。

    该环境围绕“接近目标 + 稳定停住 + 不摔倒”构建，
    采用分阶段奖励，兼顾可学习性与最终表现稳定性。
    """
    
    _cfg: VBotSection001EnvCfg
    
    def __init__(self, cfg: VBotSection001EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 目标标记和箭头
        self._target_marker_body = self._model.get_body("target_marker")
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        self._find_target_marker_dof_indices()
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        self._init_buffer()
        self._cfg.render_spacing = 0.0
    
    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        self._init_dof_pos[-self._num_action:] = self.default_angles
        # 提高动作响应，避免步态被滤波限制得过死
        self.action_filter_alpha = 0.6
    
    def _find_target_marker_dof_indices(self):
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception:
                pass
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
        
        self.num_foot_check = 4
    
    # ========== 工具函数 ==========
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def _wrap_angle(self, angle):
        angle = np.where(angle > np.pi, angle - 2*np.pi, angle)
        angle = np.where(angle < -np.pi, angle + 2*np.pi, angle)
        return angle
    
    def _euler_to_quat(self, roll, pitch, yaw):
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def _sanitize_dof_pos(self, dof_pos):
        """修复所有四元数为归一化形式"""
        sanitized = np.nan_to_num(dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 所有四元数范围
        quat_ranges = [(self._base_quat_start, self._base_quat_end)]
        if self._robot_arrow_body is not None:
            quat_ranges.append((self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end))
            quat_ranges.append((self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end))
        
        for i in range(sanitized.shape[0]):
            for quat_start, quat_end in quat_ranges:
                if quat_end <= sanitized.shape[1]:
                    quat = sanitized[i, quat_start:quat_end]
                    quat_norm = np.linalg.norm(quat)
                    if quat_norm < 1e-6 or np.isnan(quat_norm):
                        sanitized[i, quat_start:quat_end] = [0.0, 0.0, 0.0, 1.0]
                    else:
                        sanitized[i, quat_start:quat_end] = quat / quat_norm
        
        return sanitized
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """执行动作并进行轻量动作滤波。

        作用：
        1. 裁剪非法动作输入，防止异常值传入控制器。
        2. 使用一阶低通滤波降低瞬时抖动，提升步态稳定性。
        3. 计算PD力矩并写入 `actuator_ctrls`。
        """
        actions = np.clip(actions, -1.0, 1.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"].copy()
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions.copy()
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"].copy()
        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """将归一化动作映射到关节目标位，并通过PD控制输出力矩。"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)
        
        current_pos = np.nan_to_num(current_pos, nan=0.0)
        current_vel = np.nan_to_num(current_vel, nan=0.0)
        current_vel = np.clip(current_vel, -100, 100)
        
        kp = 80.0
        kv = 6.0
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _update_target_marker(self, data, pose_commands):
        """更新目标标记的可视化位置。

        注意：该函数只用于可视化，不影响奖励与终止逻辑。
        """
        num_envs = data.shape[0]
        all_dof_pos = self._sanitize_dof_pos(data.dof_pos.copy())
        
        for env_idx in range(num_envs):
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                float(pose_commands[env_idx, 0]),
                float(pose_commands[env_idx, 1]),
                float(pose_commands[env_idx, 2])
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data, robot_pos, desired_vel_xy, base_lin_vel_xy):
        """更新当前方向箭头与期望方向箭头（可视化辅助）。"""
        if self._robot_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            arrow_height = robot_pos[env_idx, 2] + 0.5
            
            # 当前方向
            cur_v = base_lin_vel_xy[env_idx]
            cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
            quat = self._euler_to_quat(0, 0, cur_yaw)
            quat = quat / max(np.linalg.norm(quat), 1e-6)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                [robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], quat
            ])
            
            # 期望方向
            des_v = desired_vel_xy[env_idx]
            des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
            quat = self._euler_to_quat(0, 0, des_yaw)
            quat = quat / max(np.linalg.norm(quat), 1e-6)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                [robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """主状态更新流程。

        执行顺序：
        1. 读取机器人状态并构造导航误差
        2. 生成期望速度命令并拼接观测
        3. 更新可视化标记
        4. 计算奖励与终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 读取状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 导航计算
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 关键判定：已到达（Phase切换门槛）
        reached_threshold = 0.5  # 50cm
        reached = distance_to_target < reached_threshold
        
        # 期望速度
        desired_vel_xy = np.clip(position_error * 0.8, -1.0, 1.0)
        desired_vel_xy = np.where(reached[:, np.newaxis], 0.0, desired_vel_xy)
        
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        desired_yaw_rate = np.clip(heading_error * 0.5, -0.5, 0.5)
        desired_yaw_rate = np.where(reached, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)
        
        # 构建观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        position_error_normalized = position_error / 10.0
        heading_error_normalized = heading_error / np.pi
        distance_normalized = np.clip(distance_to_target / 15.0, 0, 1)
        reached_flag = reached.astype(np.float32)
        
        speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        stop_ready = np.logical_and(reached, speed < 0.2)
        stop_ready_flag = stop_ready.astype(np.float32)
        
        contact_force = data.actuator_ctrls.copy()
        base_contact = np.zeros((data.shape[0], 1), dtype=np.float32)
        contact_obs = np.concatenate([contact_force, base_contact], axis=-1)
        
        obs = np.concatenate([
            noisy_linvel, noisy_gyro, projected_gravity,
            noisy_joint_angle, noisy_joint_vel, last_actions,
            command_normalized, position_error_normalized,
            heading_error_normalized[:, np.newaxis],
            distance_normalized[:, np.newaxis],
            reached_flag[:, np.newaxis],
            stop_ready_flag[:, np.newaxis],
            contact_obs,
        ], axis=-1)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新可视化
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])
        
        # 计算奖励（核心：两阶段奖励系统）
        reward = self._compute_reward(
            data, state.info, distance_to_target, reached, 
            base_lin_vel, gyro, projected_gravity, position_error
        )
        
        # 计算终止条件
        terminated = self._compute_terminated(data, root_pos, projected_gravity, reached)
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_reward(self, data, info, distance, reached, 
                        base_lin_vel, gyro, projected_gravity, position_error):
        """
        ============================================================
        【重构】基于anymal_c的速度跟踪 + 导航专用奖励
        ============================================================
        
        设计思路：
        - 先确保“会走到目标”（跟踪/接近/朝向）
        - 再确保“到点后稳定停住”（stop_bonus + 姿态稳定）
        - 同时通过倾斜预警和终止惩罚防止摔倒
        
        新结构（参考anymal_c）：
          1. 线速度跟踪奖励：鼓励跟踪期望速度
          2. 角速度跟踪奖励：鼓励跟踪角速度
          3. 接近奖励：奖励靠近目标
          4. 到达奖励：到达目标时一次性奖励
          5. 停止奖励：到达后鼓励停止（但不过度惩罚运动）
          6. 轻惩罚：出界惩罚、终止条件惩罚
        """
        num_envs = self._num_envs
        cfg = self._cfg
        
        # ===== 获取传感器信息 =====
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        
        # 获取期望速度命令（基于目标方向的简单控制律）
        # 当距离>0.5m时，向目标移动；否则停止
        position_error_norm = np.linalg.norm(position_error, axis=1, keepdims=True)
        desired_vel_xy = np.where(
            position_error_norm > 0.5,
            np.clip(position_error / (position_error_norm + 1e-6), -1.0, 1.0),  # 单位向量
            np.zeros_like(position_error)  # 到达后不需要速度
        )
        
        # ===== 1. 线速度跟踪奖励（基础步态学习） =====
        # 公式：exp(-||v_actual - v_desired||² / 0.25)
        lin_vel_error = np.sum(np.square(base_lin_vel[:, :2] - desired_vel_xy), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)
        
        # ===== 2. 角速度跟踪奖励（转向能力） =====
        # 依据“目标方向误差”构建期望偏航速度
        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        desired_yaw_rate = np.where(reached, 0.0, np.clip(1.2 * heading_error, -1.0, 1.0))
        ang_vel_error = np.square(gyro[:, 2] - desired_yaw_rate)
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        # 朝向对齐奖励（额外鼓励先转向再前进）
        heading_align_reward = np.exp(-np.square(heading_error / 0.6))
        heading_align_reward = np.where(reached, 0.0, heading_align_reward)
        
        # ===== 3. 接近奖励（鼓励持续缩短距离） =====
        # 通过历史最小距离跟踪净进展，减少来回抖动带来的虚假收益
        if "min_distance" not in info:
            info["min_distance"] = distance.copy()
        
        distance_improvement = info["min_distance"] - distance
        info["min_distance"] = np.minimum(info["min_distance"], distance)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # 每接近1米得4分
        
        # ===== 4. 到达奖励（一次性触发） =====
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached)
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)
        
        # 前向投影速度奖励：鼓励沿目标方向迈步，而非原地抖动
        target_dir = position_error / (position_error_norm + 1e-6)
        forward_speed = np.sum(base_lin_vel[:, :2] * target_dir, axis=1)
        forward_progress_reward = np.where(reached, 0.0, np.clip(forward_speed, -0.5, 1.5))

        # ===== 5. 停止奖励（到达后鼓励低速稳定） =====
        # 仅在 reached 阶段生效；速度越低、偏航越稳，奖励越高
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        stop_base = 2.0 * (0.8 * np.exp(-(speed_xy / 0.2)**2) + 1.2 * np.exp(-(np.abs(gyro[:, 2]) / 0.1)**4))
        stop_bonus = np.where(reached, stop_base, 0.0)
        
        # ===== 6. 惩罚项 =====
        
        # 终止条件惩罚
        termination_penalty = np.zeros(num_envs, dtype=np.float32)
        
        # 检查DOF速度超限
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > cfg.max_dof_vel
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)
        
        # 倾斜惩罚（高角度直接强惩）
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)
        
        # 轻微惩罚项（参考anymal_c）
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])  # Z轴运动惩罚
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)  # 侧倾惩罚
        
        # 姿态稳定性奖励：主动鼓励保持竖直（防摔倒关键）
        # projected_gravity 在正常站立时应该 ≈ [0, 0, -1]
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1]) + np.square(projected_gravity[:, 2] + 1.0)
        
        # 提前预警倾斜：25°开始递增惩罚，避免“临界角才纠正”
        tilt_warning = np.where(
            tilt_angle > np.deg2rad(25),
            -8.0 * (tilt_angle / np.deg2rad(60)) ** 2,  # 25°时约-1.4，60°时-8.0
            0.0
        )
        
        # ===== 综合奖励 =====
        # 两阶段奖励结构：reached / not reached
        reward = np.where(
            reached,
            # 已到达：主要是停止奖励
            (
                stop_bonus +
                arrival_bonus +
                - 1.0 * lin_vel_z_penalty +
                - 0.1 * ang_vel_xy_penalty +
                - 0.5 * orientation_penalty +   # 姿态稳定（防摔倒）权重增强
                tilt_warning +                  # 提前预警倾斜
                termination_penalty
            ),
            # 未到达：速度跟踪 + 接近奖励
            (
                1.2 * tracking_lin_vel +       # 线速度跟踪权重（略降，避免过拘束）
                0.8 * tracking_ang_vel +       # 角速度跟踪权重（提高，强化转向学习）
                0.6 * heading_align_reward +   # 朝向对齐奖励
                0.6 * forward_progress_reward +# 朝目标前进奖励
                approach_reward +              # 接近奖励
                - 1.0 * lin_vel_z_penalty +
                - 0.1 * ang_vel_xy_penalty +
                - 0.5 * orientation_penalty +   # 姿态稳定（防摔倒）权重增强
                tilt_warning +                  # 提前预警倾斜
                termination_penalty
            )
        )
        
        # 奖励裁剪防止数值溢出
        reward = np.clip(reward, -50.0, 20.0)
        reward = np.nan_to_num(reward, nan=0.0)
        
        return reward
    
    def _compute_terminated(self, data, root_pos, projected_gravity, reached):
        """
        ============================================================
        【终止条件】三个关键场景
        ============================================================
        
        1. 摔倒：倾斜角超过阈值
        2. 出界：距离中心超边界
        3. 基座触地：非法接触
        4. 数值发散：关节速度异常
        """
        terminated = np.zeros(self._num_envs, dtype=bool)
        
        # 1. 摔倒（倾斜 > 60°）
        tilt = np.linalg.norm(projected_gravity[:, :2], axis=1)
        tilt_angle = np.arctan2(tilt, np.abs(projected_gravity[:, 2]))
        fallen = tilt_angle > np.deg2rad(60)  # sin(60°) ≈ 0.866
        terminated |= fallen
        
        # 2. 出界 (距离 > 12.5m)
        distance_from_center = np.linalg.norm(root_pos[:, :2], axis=1)
        boundary_radius = 12.5
        out_of_bounds = distance_from_center > boundary_radius
        terminated |= out_of_bounds
        
        # 3. 基座碰撞地面
        try:
            if self.num_termination_check > 0:
                cquerys = self._model.get_contact_query(data)
                term_check = cquerys.is_colliding(self.termination_contact)
                term_check = term_check.reshape((self._num_envs, self.num_termination_check))
                base_collision = term_check.any(axis=1)
                terminated |= base_collision
        except Exception:
            pass
        
        # 4. 物理发散 (异常值检测)
        try:
            joint_vel = self.get_dof_vel(data)
            vel_extreme = np.abs(joint_vel).max(axis=1) > 1e6
            terminated |= vel_extreme
        except Exception:
            pass
        
        return terminated
    
    # ========== 重置 ==========
    
    def _generate_circle_ring_points(self, num_points, radius_min, radius_max, center=(0.0, 0.0)):
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(radius_min, radius_max, num_points)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        return np.column_stack([x, y])
    
    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """重置环境并构建初始观测/信息字典。

        关键步骤：
        1. 在环形区域随机出生位置
        2. 使用随机 yaw（360°）初始化朝向
        3. 初始化导航目标（中心点）与缓存信息
        """
        cfg = self._cfg
        num_envs = data.shape[0]
        
        # 出生位置
        robot_init_xy = self._generate_circle_ring_points(
            num_envs,
            radius_min=cfg.init_state.spawn_ring_radius_min,
            radius_max=cfg.init_state.spawn_ring_radius_max,
            center=(0.0, 0.0)
        )
        robot_init_xyz = np.column_stack([robot_init_xy, np.full(num_envs, 0.5)])
        
        # DOF设置
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        dof_pos[:, 3:6] = robot_init_xyz
        
        # 随机初始朝向：360度随机 yaw
        random_yaw = np.random.uniform(-np.pi, np.pi, num_envs)
        for i in range(num_envs):
            # 机器人body：随机yaw的四元数
            quat = self._euler_to_quat(0.0, 0.0, random_yaw[i])  # roll=0, pitch=0, yaw=random
            dof_pos[i, self._base_quat_start:self._base_quat_end] = quat
            if self._robot_arrow_body is not None:
                dof_pos[i, self._robot_arrow_dof_start + 3:self._robot_arrow_dof_end] = quat
                dof_pos[i, self._desired_arrow_dof_start + 3:self._desired_arrow_dof_end] = [0.0, 0.0, 0.0, 1.0]
        
        # 目标：中心(0,0)
        target_positions = np.zeros((num_envs, 2), dtype=np.float32)
        target_headings = np.zeros((num_envs, 1), dtype=np.float32)
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 应用
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        self._update_target_marker(data, pose_commands)
        
        # 计算初始观测
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        robot_position = root_pos[:, :2]
        target_position = pose_commands[:, :2]
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        
        reached = distance_to_target < 0.5
        
        desired_vel_xy = np.clip(position_error * 0.8, -1.0, 1.0)
        desired_yaw_rate = np.clip(heading_error * 0.5, -0.5, 0.5)
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)
        
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        
        position_error_normalized = position_error / 10.0
        heading_error_normalized = heading_error / np.pi
        distance_normalized = np.clip(distance_to_target / 15.0, 0, 1)
        reached_flag = reached.astype(np.float32)
        stop_ready_flag = np.zeros(num_envs, dtype=np.float32)
        
        contact_force = np.zeros((num_envs, 12), dtype=np.float32)
        base_contact = np.zeros((num_envs, 1), dtype=np.float32)
        contact_obs = np.concatenate([contact_force, base_contact], axis=-1)
        
        obs = np.concatenate([
            noisy_linvel, noisy_gyro, projected_gravity,
            noisy_joint_angle, noisy_joint_vel, last_actions,
            command_normalized, position_error_normalized,
            heading_error_normalized[:, np.newaxis],
            distance_normalized[:, np.newaxis],
            reached_flag[:, np.newaxis],
            stop_ready_flag[:, np.newaxis],
            contact_obs,
        ], axis=-1)
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "prev_distance": distance_to_target.copy(),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
        }
        
        return obs, info
