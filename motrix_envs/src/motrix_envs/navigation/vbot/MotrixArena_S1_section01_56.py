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

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection01EnvCfg


@registry.env("MotrixArena_S1_section01_56", "np")
@registry.env("MotrixArena_S1_section011_56", "np")
@registry.env("MotrixArena_S1_section012_56", "np")
@registry.env("MotrixArena_S1_section013_56", "np")

class VBotSection01Env(NpEnv):
    """
    VBot在Section01地形上的导航任务
    继承自NpEnv，使用VBotSection01EnvCfg配置
    """
    _cfg: VBotSection01EnvCfg
    
    def __init__(self, cfg: VBotSection01EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：67维（55 + 12维接触力）
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_xy_range = np.array(cfg.init_state.pos_randomization_range, dtype=np.float32)
    
        # 导航统计计数器
        self.navigation_stats_step = 0

        # ========== 课程学习（Curriculum Learning） ==========
        # 50%环境从正常起点出发，50%随机散布在赛道上
        # 让机器人从近处学起，逐步掌握全程
        self._curriculum_spawn_fraction = 0.5  # 50%环境用课程出生
        self._curriculum_spawn_z = 1.5  # 课程出生高度，足够高避免卡地形

        # ========== 从cfg读取阶段任务参数 ==========
        task_cfg = cfg.task_config
        self.task_name = task_cfg.task_name
        self.enable_landmark_rewards = task_cfg.enable_landmark_rewards
        self.enable_celebration_reward = task_cfg.enable_celebration_reward

        self.smile_positions = np.array(task_cfg.smile_positions, dtype=np.float32)
        if self.smile_positions.size == 0:
            self.smile_positions = np.zeros((0, 2), dtype=np.float32)
        self.smile_radius = float(task_cfg.smile_radius)
        self.smile_reward = float(task_cfg.smile_reward)

        self.package_positions = np.array(task_cfg.package_positions, dtype=np.float32)
        if self.package_positions.size == 0:
            self.package_positions = np.zeros((0, 2), dtype=np.float32)
        self.package_radius = float(task_cfg.package_radius)
        self.package_reward = float(task_cfg.package_reward)

        self.goal_y = float(task_cfg.goal_y)
        self.goal_reached_reward = float(task_cfg.goal_reached_reward)

        self.celebration_reward = float(task_cfg.celebration_reward)
        self.required_jumps = int(task_cfg.required_jumps)

        self.boundary_x = float(task_cfg.boundary_x)
        self.boundary_y_max = float(task_cfg.boundary_y_max)
        self.tilt_threshold_deg = float(task_cfg.tilt_threshold_deg)
        self.gravity_z_termination_threshold = -np.cos(np.deg2rad(self.tilt_threshold_deg))
    
        # ========== 硬编码导航调试模式 ==========
        # 设为True可让机器人按预定顺序访问所有地标点，用于测试最优路径和奖励上界
        self.use_hardcoded_navigation = True  # 改为True时启用
        self._build_hardcoded_waypoints()
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
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
        self.action_filter_alpha = 0.6  # v7.2: 与section001保持一致，加快响应
    
    def _build_hardcoded_waypoints(self):
        """构建硬编码路径：START → 左→中→右笑脸 → 右→中→左红包 → 终点
        之字形路径，减少来回往返距离
        """
        waypoints = []
        # 起点
        waypoints.append(np.array([0.0, -2.4], dtype=np.float32))  # START
        
        # 3个笑脸：按顺序访问（已排序）
        for smile in self.smile_positions:
            waypoints.append(np.array(smile, dtype=np.float32))
        
        # 3个红包：改为反向访问（从右到左）形成之字形
        # 原始顺序: [-3, 4.1], [0, 4.1], [3, 4.1]
        # 反向顺序: [3, 4.1], [0, 4.1], [-3, 4.1]
        reversed_packages = [self.package_positions[2], self.package_positions[1], self.package_positions[0]]
        for pkg in reversed_packages:
            waypoints.append(np.array(pkg, dtype=np.float32))
        
        # 终点：只需y=8即可，x可以任意（这里用0作为中心）
        waypoints.append(np.array([0.0, self.goal_y], dtype=np.float32))  # GOAL
        self._hardcoded_waypoints = np.array(waypoints, dtype=np.float32)
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
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
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        
        # 获取所有地面geom（遍历所有geom，找到包含ground_subtree名称的）
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree  # "0ground_root"
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        # if len(ground_geoms) == 0:
        #     print(f"[Warning] 未找到以 '{ground_prefix}' 开头的地面geom！")
        #     self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        #     self.num_termination_check = 0
        #     return
        
        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            print(f"[Info] 初始化终止接触检测: {len(termination_contact_names)}个基座geom × {len(ground_geoms)}个地面geom = {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4  
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
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
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
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
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # ========== 硬编码导航覆盖路点 ==========
        if self.use_hardcoded_navigation:
            if "hardcoded_waypoint_idx" not in state.info:
                state.info["hardcoded_waypoint_idx"] = np.zeros(data.shape[0], dtype=np.int32)
            
            robot_xy = root_pos[:, :2]
            current_waypoints = self._hardcoded_waypoints[state.info["hardcoded_waypoint_idx"]]
            
            # 检测是否到达当前路点（距离<0.5m）
            dist_to_wp = np.linalg.norm(robot_xy - current_waypoints, axis=1)
            reached_wp = dist_to_wp < 0.5
            
            # 更新到下一路点（不超过最后一个）
            max_wp_idx = len(self._hardcoded_waypoints) - 1
            state.info["hardcoded_waypoint_idx"] = np.minimum(
                state.info["hardcoded_waypoint_idx"] + reached_wp.astype(np.int32),
                max_wp_idx
            )
            
            next_waypoints = self._hardcoded_waypoints[np.minimum(
                state.info["hardcoded_waypoint_idx"],
                max_wp_idx
            )]
            pose_commands = np.column_stack([next_waypoints, np.zeros(data.shape[0], dtype=np.float32)])
            state.info["pose_commands"] = pose_commands
        else:
            pose_commands = state.info["pose_commands"]
        
        # 导航目标
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定（只看位置，与奖励计算保持一致）
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令（与平地navigation一致，简单P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向（从当前位置指向目标）
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
        # 计算接触力观测（12个关节 + 1个基座接触 = 13维）
        # 从关节力矩估计接触力（简化：用actuator_ctrls代表接触效应）
        contact_force = data.actuator_ctrls.copy()  # 12维（12个执行器控制）
        # 添加基座接触标志（1维）
        base_contact = np.zeros((data.shape[0], 1), dtype=np.float32)
        contact_obs = np.concatenate([contact_force, base_contact], axis=-1)  # 13维
        
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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                contact_obs,        # 13维（12个关节 + 1个基座）
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 67)  # 54 + 13 = 67维
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
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
        
        return state
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        终止条件:
        1. 真正翻倒 (倾斜>70°, 坑洼中30-70°倾斜是正常的)
        2. 超出边界 (X>±6m 或 Y>各阶段设定)
        3. 物理发散 (NaN检测)
        4. 卡住不动 (连续400步速度<0.05且未到达终点)
        """
        data = state.data
        info = state.info
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # 获取projected gravity来检测倾斜
        projected_gravity = self._compute_projected_gravity(root_quat)
        gravity_z = projected_gravity[:, 2]  # Z分量，正常站立时应≈-1
        
        # 倾斜检测：cos(60°) ≈ 0.5，gravity_z应该<-0.5才算正常
        tilt_terminated = gravity_z > self.gravity_z_termination_threshold
        
        # 边界检测（X方向有墙壁不需要检测；Y<-3.5掉下悬崖；Y超出地图终止）
        # 出生点Y=-2.4±0.5，最远Y=-2.9，所以终止线放-3.5给足缓冲
        y_out_forward = root_pos[:, 1] > self.boundary_y_max
        y_fall_backward = root_pos[:, 1] < -3.5  # 后退掉落悬崖（远离出生点）
        boundary_terminated = y_out_forward | y_fall_backward
        
        # 物理发散检测
        nan_terminated = np.any(np.isnan(root_pos), axis=1)
        
        # 卡住检测：低速持续时间
        if "stuck_counter" not in info:
            info["stuck_counter"] = np.zeros(root_pos.shape[0], dtype=np.int32)
        
        # 检测XY平面速度
        xy_speed = np.linalg.norm(root_vel[:, :2], axis=1)
        is_moving = xy_speed > 0.05  # 速度阈值5cm/s
        
        # 更新卡住计数器
        info["stuck_counter"] = np.where(is_moving, 0, info["stuck_counter"] + 1)
        
        # 卡住超过400步（4秒）且未到达终点就终止
        reached_goal = root_pos[:, 1] >= self.goal_y
        stuck_terminated = (info["stuck_counter"] > 400) & (~reached_goal)
        
        # 合并所有终止条件
        terminated = tilt_terminated | boundary_terminated | nan_terminated | stuck_terminated
        
        return state.replace(terminated=terminated, info=info)
    
    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        Phase1 (Section011) 导航任务奖励计算 - 优化路径引导
        
        最优路线: START → 笑脸层横向收集 → 红包层横向收集 → 2026平台 → 庆祝
        得分系统:
        - 笑脸收集: 3个×4分 = 12分
        - 红包收集: 3个×2分 = 6分
        - 到达2026: 20分
        - 庆祝动作: 2分
        总计最高40分
        """
        cfg = self._cfg
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        robot_xy = root_pos[:, :2]
        robot_y = root_pos[:, 1]
        
        num_envs = data.shape[0]
        reward = np.zeros(num_envs, dtype=np.float32)
        
        # ========== 1. 阶段性引导奖励 ==========
        # 统计已收集的地标数量
        smile_collected_count = np.zeros(num_envs, dtype=np.int32)
        package_collected_count = np.zeros(num_envs, dtype=np.int32)
        
        if self.enable_landmark_rewards:
            for i in range(len(self.smile_positions)):
                collected_key = f"smile_{i}_collected"
                if collected_key in info:
                    smile_collected_count += info[collected_key].astype(np.int32)
            
            for i in range(len(self.package_positions)):
                collected_key = f"package_{i}_collected"
                if collected_key in info:
                    package_collected_count += info[collected_key].astype(np.int32)
        
        # ========== 专家级奖励系统 v7.5 ==========
        # v7.5修复：机器人默认朝+X，但目标在+Y → v7.4硬编码vy导致方向错误
        # 关键改动：前进奖励改为dot(vel, goal_dir)投影，加回弱朝向对齐
        # 保持RL自主决策：速度大小自由 + 地标收集自然引导横向
        
        # --- 计算目标方向（用于前进投影+距离势函数） ---
        pose_commands = info["pose_commands"]
        goal_xy = pose_commands[:, :2]
        goal_dir = goal_xy - robot_xy
        goal_dist = np.linalg.norm(goal_dir, axis=1, keepdims=True)
        goal_dir_unit = goal_dir / np.maximum(goal_dist, 1e-6)
        
        # 1. 速度大小跟踪（步态核心驱动力：跑快就奖励，不限方向）
        vel_xy = root_vel[:, :2]
        speed = np.linalg.norm(vel_xy, axis=1)  # 速度大小，任意方向
        speed_error = np.square(speed - 1.0)  # 期望速度1.0m/s
        tracking_speed = np.exp(-speed_error / 0.25)
        reward += 1.0 * tracking_speed  # 跑得快就有奖励，不管往哪跑
        
        # 2. 前进投影（v7.5修复：从section001移植dot(vel, target_dir)，不再硬编码vy）
        # 机器人朝+X但目标在+Y，用投影才能正确奖励朝目标方向的速度
        vel_toward_goal = np.sum(vel_xy * goal_dir_unit.squeeze(), axis=1)
        forward_progress = np.clip(vel_toward_goal, -0.5, 1.5)
        reward += 0.8 * forward_progress  # 权重提高到0.8，方向正确了可以给更多
        
        # 3. 弱朝向对齐（帮助初始转向：机器人朝+X需要转90°朝+Y）
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(goal_dir[:, 1], goal_dir[:, 0]).flatten()
        heading_error = desired_heading - robot_heading
        heading_error = np.where(heading_error > np.pi, heading_error - 2*np.pi, heading_error)
        heading_error = np.where(heading_error < -np.pi, heading_error + 2*np.pi, heading_error)
        heading_align = np.exp(-np.square(heading_error / 0.6))
        reward += 0.3 * heading_align  # 弱朝向，不压制横向收集地标
        
        # v7.6: 移除角速度限制，让机器人可以自由转动
        # 卡住时多转转可能找到新路径，不应该限制yaw_rate
        
        # 4. 距离终点势函数（靠近终点就奖励，保持全局前进趋势）
        dist_to_goal = goal_dist.flatten()
        if "prev_dist_to_goal" not in info:
            info["prev_dist_to_goal"] = dist_to_goal.copy()
        dist_improvement = info["prev_dist_to_goal"] - dist_to_goal
        goal_shaping = np.clip(dist_improvement * 2.0, -0.1, 0.2)
        reward += goal_shaping
        info["prev_dist_to_goal"] = dist_to_goal.copy()
        
        # 6. 存活奖励（鼓励不摔倒）
        reward += 0.005
        
        # 7. 新区域探索奖励（到达距目标更近的位置时给一次性奖励）
        if "min_dist_to_goal" not in info:
            info["min_dist_to_goal"] = dist_to_goal.copy()
        new_territory = dist_to_goal < info["min_dist_to_goal"] - 0.2  # 比之前最近再近0.5m
        reward += new_territory.astype(np.float32) * 1.0  # 每次突破+1.0
        info["min_dist_to_goal"] = np.minimum(info["min_dist_to_goal"], dist_to_goal)
        
        # 8. 软倾斜惩罚（渐进式：坑洼地形倾斜是正常的，不应终止）
        projected_gravity = self._compute_projected_gravity(root_quat)
        gravity_z = projected_gravity[:, 2]  # 正常站立时≈-1，完全倒下时≈0
        # tilt_angle ≈ arccos(-gravity_z)：0°=站直，90°=侧翻
        tilt_cos = np.clip(-gravity_z, 0.0, 1.0)  # 0=倒下, 1=站直
        # 45°以内无惩罚，45-70°渐进惩罚，70°以上终止(由cfg.tilt_threshold_deg控制)
        # v7.1: 30°→45°，快跑过坑洼时30-40°倾斜很常见，不应惩罚
        soft_tilt_penalty = np.where(
            tilt_cos > 0.707,  # cos(45°)=0.707, tilt<45°: 无惩罚
            0.0,
            -0.002 * (0.707 - tilt_cos) / 0.707  # 最大约-0.002/步
        )
        reward += soft_tilt_penalty
        
        # 2. 地标磁吸引力（v7.5修复：用dot(vel, dir)代替仅X方向吸引）
        landmark_attraction = np.zeros(num_envs, dtype=np.float32)
        
        if self.enable_landmark_rewards:
            vel_2d = root_vel[:, :2]  # [E, 2] XY速度
            
            # 笑脸吸引力（全向量化）- v7.7增强版：让策略主动收集
            if len(self.smile_positions) > 0:
                s_diff = self.smile_positions[np.newaxis, :, :] - robot_xy[:, np.newaxis, :]  # [E, N_s, 2]
                s_dist = np.linalg.norm(s_diff, axis=2)  # [E, N_s]
                s_dir = s_diff / np.maximum(s_dist[:, :, np.newaxis], 1e-6)  # [E, N_s, 2] 单位方向
                
                s_collected = np.stack([
                    info.get(f"smile_{i}_collected", np.zeros(num_envs, dtype=bool))
                    for i in range(len(self.smile_positions))
                ], axis=1)
                
                s_valid = (s_dist < 6.0) & (~s_collected)  # 检测范围4m→6m，更早感知
                s_strength = np.clip((6.0 - s_dist) / 6.0, 0, 1) * s_valid
                # dot(vel, dir): 向地标方向移动就奖励
                s_vel_toward = np.sum(vel_2d[:, np.newaxis, :] * s_dir, axis=2)  # [E, N_s]
                landmark_attraction += np.sum(
                    s_strength * np.clip(s_vel_toward * 0.01, 0, 0.02), axis=1
                )
            
            # 红包吸引力（全向量化）
            if len(self.package_positions) > 0:
                p_diff = self.package_positions[np.newaxis, :, :] - robot_xy[:, np.newaxis, :]
                p_dist = np.linalg.norm(p_diff, axis=2)
                p_dir = p_diff / np.maximum(p_dist[:, :, np.newaxis], 1e-6)
                
                p_collected = np.stack([
                    info.get(f"package_{i}_collected", np.zeros(num_envs, dtype=bool))
                    for i in range(len(self.package_positions))
                ], axis=1)
                
                p_valid = (p_dist < 4.0) & (~p_collected)
                p_strength = np.clip((4.0 - p_dist) / 4.0, 0, 1) * p_valid
                p_vel_toward = np.sum(vel_2d[:, np.newaxis, :] * p_dir, axis=2)
                landmark_attraction += np.sum(
                    p_strength * np.clip(p_vel_toward * 0.01, 0, 0.02), axis=1
                )
        
        reward += landmark_attraction
        
        # 3. 卡住时鼓励旋转探索（v7.6: 横移→旋转，遇到洼地转个方向找出路）
        # 当stuck_counter>100步(1秒)时，奖励角速度，鼓励转向寻找新路径
        stuck_counter = info.get("stuck_counter", np.zeros(num_envs, dtype=np.int32))
        is_stuck = stuck_counter > 100
        angular_speed = np.abs(gyro[:, 2])  # 角速度绝对值（转得越快越好）
        rotation_explore_reward = np.where(
            is_stuck,
            np.clip(angular_speed * 0.8, 0, 0.4),  # 卡住时转动有奖励
            0.0
        )
        reward += rotation_explore_reward
        
        # 4. 后退警告惩罚（Y方向接近起点后方悬崖时惩罚）
        # X方向有墙壁物理屏障，不需要软惩罚
        spawn_y = self.spawn_center[1]  # -2.4
        y_backward_dist = robot_y - (spawn_y - 1.0)  # 距离悬崖边缘的距离
        too_close_backward = y_backward_dist < 1.0  # 1m安全距离
        backward_penalty = np.where(
            too_close_backward,
            -0.5 * (1.0 - y_backward_dist),  # 每步最大-0.5
            0.0
        )
        reward += backward_penalty
        
        # ========== 2. 地标收集奖励（区域化检测） ==========
        if self.enable_landmark_rewards:
            # 笑脸：使用矩形区域 ±1.0m（增大检测范围，促进早期探索发现）
            for i, smile_pos in enumerate(self.smile_positions):
                # 区域判断：|X - lx| < 1.0 且 |Y - ly| < 1.0
                in_x_range = np.abs(robot_xy[:, 0] - smile_pos[0]) < 1.0
                in_y_range = np.abs(robot_xy[:, 1] - smile_pos[1]) < 1.0
                in_range = in_x_range & in_y_range

                collected_key = f"smile_{i}_collected"
                if collected_key not in info:
                    info[collected_key] = np.zeros(num_envs, dtype=bool)

                first_collect = in_range & (~info[collected_key])
                reward += first_collect.astype(np.float32) * self.smile_reward
                info[collected_key] = info[collected_key] | in_range

            # 红包：使用矩形区域 ±0.6m（增大检测范围）
            for i, pkg_pos in enumerate(self.package_positions):
                # 区域判断：|X - lx| < 0.6 且 |Y - ly| < 0.6
                in_x_range = np.abs(robot_xy[:, 0] - pkg_pos[0]) < 0.6
                in_y_range = np.abs(robot_xy[:, 1] - pkg_pos[1]) < 0.6
                in_range = in_x_range & in_y_range

                collected_key = f"package_{i}_collected"
                if collected_key not in info:
                    info[collected_key] = np.zeros(num_envs, dtype=bool)

                first_collect = in_range & (~info[collected_key])
                reward += first_collect.astype(np.float32) * self.package_reward
                info[collected_key] = info[collected_key] | in_range

        # ========== 3.5 收集率优化（Coverage Ratio Shaping） ==========
        # 目标：鼓励探索多个笑脸/红包区域，而不是只冲终点
        collection_ratio = np.zeros(num_envs, dtype=np.float32)
        if self.enable_landmark_rewards:
            total_smiles = len(self.smile_positions)
            total_packages = len(self.package_positions)
            total_landmarks = total_smiles + total_packages

            if total_landmarks > 0:
                smile_collected_count = np.zeros(num_envs, dtype=np.int32)
                package_collected_count = np.zeros(num_envs, dtype=np.int32)

                for i in range(total_smiles):
                    smile_collected_count += info.get(f"smile_{i}_collected", np.zeros(num_envs, dtype=bool)).astype(np.int32)
                for i in range(total_packages):
                    package_collected_count += info.get(f"package_{i}_collected", np.zeros(num_envs, dtype=bool)).astype(np.int32)

                collected_total = smile_collected_count + package_collected_count
                collection_ratio = collected_total.astype(np.float32) / float(total_landmarks)

                # 只奖励“收集率的提升”（避免无意义停留刷分）
                if "prev_collection_ratio" not in info:
                    info["prev_collection_ratio"] = collection_ratio.copy()
                ratio_delta = collection_ratio - info["prev_collection_ratio"]
                coverage_progress_reward = np.clip(ratio_delta * 6.0, -0.05, 0.5)
                reward += coverage_progress_reward
                info["prev_collection_ratio"] = collection_ratio.copy()

                # 在到达终点前，给一个轻量密集偏置，持续鼓励更高收集率
                reward += np.where(robot_y < self.goal_y, 0.02 * collection_ratio, 0.0)
        
        # ========== 4. 到达终点奖励（强制完整收集） ==========
        reached_goal = robot_y >= self.goal_y
        
        goal_reached_key = "goal_reached"
        if goal_reached_key not in info:
            info[goal_reached_key] = np.zeros(num_envs, dtype=bool)
        
        first_reach = reached_goal & (~info[goal_reached_key])
        
        # 到达终点奖励按收集率加权：
        # 收集率0% -> 16%基础分；收集率100% -> 100%满分
        if self.enable_landmark_rewards:
            bonus_reward = self.goal_reached_reward * (0.16 + 0.84 * collection_ratio)
        else:
            bonus_reward = np.full(num_envs, self.goal_reached_reward, dtype=np.float32)
        
        reward += first_reach.astype(np.float32) * bonus_reward
        info[goal_reached_key] = info[goal_reached_key] | reached_goal
        
        # ========== 5. 庆祝动作检测（原地旋转360°） ==========
        # 最简单稳定的庆祝：到达平台后原地转圈，检测累计旋转角度
        if "celebration_yaw_start" not in info:
            info["celebration_yaw_start"] = np.zeros(num_envs, dtype=np.float32)
        if "celebration_yaw_accumulated" not in info:
            info["celebration_yaw_accumulated"] = np.zeros(num_envs, dtype=np.float32)
        if "celebration_prev_yaw" not in info:
            info["celebration_prev_yaw"] = robot_heading.copy()
        if "celebration_done" not in info:
            info["celebration_done"] = np.zeros(num_envs, dtype=bool)

        at_goal = info[goal_reached_key]
        
        # 到达终点时记录初始朝向
        just_reached = at_goal & (~info["celebration_done"]) & (info["celebration_yaw_accumulated"] == 0)
        info["celebration_yaw_start"] = np.where(just_reached, robot_heading, info["celebration_yaw_start"])
        
        # 计算本步旋转角度（处理-π到π的跳变）
        yaw_delta = robot_heading - info["celebration_prev_yaw"]
        yaw_delta = np.where(yaw_delta > np.pi, yaw_delta - 2*np.pi, yaw_delta)
        yaw_delta = np.where(yaw_delta < -np.pi, yaw_delta + 2*np.pi, yaw_delta)
        
        # 累计旋转（只在到达终点后计数，离开终点重置）
        info["celebration_yaw_accumulated"] = np.where(
            at_goal,
            info["celebration_yaw_accumulated"] + np.abs(yaw_delta),
            0.0
        )
        info["celebration_prev_yaw"] = robot_heading.copy()
        
        # 过程奖励：到平台后鼓励持续旋转（角速度越大奖励越高）
        angular_speed = np.abs(gyro[:, 2])
        reward += np.where(at_goal & (~info["celebration_done"]), 
                          np.clip(angular_speed * 0.05, 0.0, 0.05), 
                          0.0)
        
        # 完成奖励：累计旋转超过required_jumps圈（默认3圈=6π弧度）
        if self.enable_celebration_reward and self.required_jumps > 0:
            required_rotation = self.required_jumps * 2 * np.pi  # 3圈
            celebration_complete = (
                (info["celebration_yaw_accumulated"] >= required_rotation)
                & (~info["celebration_done"])
                & at_goal
            )
            reward += celebration_complete.astype(np.float32) * self.celebration_reward
            info["celebration_done"] = info["celebration_done"] | celebration_complete
        
        # ========== 5. 稳定性惩罚 ==========
        # v7.1: 移除角速度惩罚——大步快跑时陀螺仪值大是正常的
        # 已有倾斜检测(>70°终止)防止摔倒，不需要额外惩罚角速度
        
        # 终止惩罚（仅对真正翻倒>70°才给，坑洼中倾斜不惩罚）
        # 注意：soft_tilt_penalty已经在上面处理了45-70°的渐进惩罚
        tilt_terminated = gravity_z > self.gravity_z_termination_threshold  # 复用上面的gravity_z
        reward += tilt_terminated.astype(np.float32) * (-10.0)
        
        # 后退掉落惩罚（Y<-3.5掉下悬崖，与终止条件一致）
        y_fall = robot_y < -3.5
        reward += y_fall.astype(np.float32) * (-30.0)
        
        return reward
    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection01EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # 按配置中的随机范围生成初始位置偏移
        # pos_randomization_range: [x_min, y_min, x_max, y_max]
        random_xy = np.random.uniform(
            low=self.spawn_xy_range[:2],
            high=self.spawn_xy_range[2:4],
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_xy  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        
        # === 课程学习：50%环境随机散布在赛道上 ===
        # 让机器人先学近处（靠近终点），再逐步掌握远处（起点到终点全程）
        num_curriculum = int(num_envs * self._curriculum_spawn_fraction)
        if num_curriculum > 0:
            cur_idx = np.arange(num_curriculum)  # 前一半环境用课程出生
            # Y: 从起点到终点前1.5m范围内随机
            cur_y = np.random.uniform(
                self.spawn_center[1], self.goal_y - 1.5,
                size=num_curriculum
            )
            # X: 赛道宽度内随机（避开墙壁）
            cur_x = np.random.uniform(-4.0, 4.0, size=num_curriculum)
            robot_init_xy[cur_idx, 0] = cur_x
            robot_init_xy[cur_idx, 1] = cur_y
            terrain_heights[cur_idx] = self._curriculum_spawn_z  # 高处出生，安全落地
        
        # 组合XYZ坐标
        robot_init_pos = robot_init_xy  # [num_envs, 2]
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz  # [x, y, z] 随机生成的位置
        
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2],
            high=cfg.commands.pose_command_range[3:5],
            size=(num_envs, 2)
        )
        # 目标在cfg中定义为世界坐标，直接使用
        target_positions = target_offset
        
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1)
        )
        
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 归一化base的四元数（DOF 6-9）
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
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
        
        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 计算速度命令
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只看位置
        
        # 计算期望速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # ===== 与reset一致：角速度跟踪运动方向 =====
        # 计算期望的运动方向（从update_state中复制）
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        
        # 添加死区，与update_state保持一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        # 计算接触力观测（12个关节 + 1个基座接触 = 13维）
        contact_force = np.zeros((num_envs, 12), dtype=np.float32)  # 12维
        base_contact = np.zeros((num_envs, 1), dtype=np.float32)  # 1维
        contact_obs = np.concatenate([contact_force, base_contact], axis=-1)  # 13维

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
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                contact_obs,        # 13维（12个关节 + 1个基座）
            ],
            axis=-1,
        )
        print(f"obs.shape:{obs.shape}")
        assert obs.shape == (num_envs, 67)  # 54 + 13 = 67维
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 统一使用min_distance机制
            # 新增：与locomotion一致的字段
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 上一步关节速度
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),  # 足部接触状态
            "goal_reached": np.zeros(num_envs, dtype=bool),
            "celebration_yaw_start": np.zeros(num_envs, dtype=np.float32),
            "celebration_yaw_accumulated": np.zeros(num_envs, dtype=np.float32),
            "celebration_prev_yaw": robot_heading.copy(),
            "celebration_done": np.zeros(num_envs, dtype=bool),
            "prev_dist_to_goal": np.linalg.norm(
                pose_commands[:, :2] - root_pos[:, :2], axis=1
            ),
            "min_dist_to_goal": np.linalg.norm(
                pose_commands[:, :2] - root_pos[:, :2], axis=1
            ),
            "prev_collection_ratio": np.zeros(num_envs, dtype=np.float32),
        }

        # 动态初始化地标收集状态，便于不同阶段复用
        for i in range(len(self.smile_positions)):
            info[f"smile_{i}_collected"] = np.zeros(num_envs, dtype=bool)
        for i in range(len(self.package_positions)):
            info[f"package_{i}_collected"] = np.zeros(num_envs, dtype=bool)
        
        # 硬编码导航初始化
        if self.use_hardcoded_navigation:
            info["hardcoded_waypoint_idx"] = np.zeros(num_envs, dtype=np.int32)
        
        return obs, info
    