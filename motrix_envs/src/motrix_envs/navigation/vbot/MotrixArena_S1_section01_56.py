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
        
        # 关键指标追踪（用于诊断）
        self.metric_window = {
            'robot_y': [],
            'rewards': [],
            'stair_pass': [],
            'stuck_ratio': [],
            'max_waypoint': []
        }
        self.metric_window_size = 1000

        # ========== 课程学习（Curriculum Learning） ==========
        # 50%环境从正常起点出发，50%随机散布在赛道上
        # 让机器人从近处学起，逐步掌握全程
        self._curriculum_spawn_fraction = 0.5  # 50%环境用课程出生
        self._curriculum_spawn_z = 2.0  # 课程出生高度（2.0m），足够高避免卡地形、吊桥、河床等障碍

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
        self._milestone_positions = np.array(task_cfg.milestone_positions, dtype=np.float32)
        if self._milestone_positions.size == 0:
            self._milestone_positions = np.array([[-3.0, 8.0], [0.0, 24.0], [0.0, 32.0]], dtype=np.float32)
        self._milestone_stay_radius = 1.0
        self._milestone_hold_reward = 0.02
        self._milestone_still_reward = 0.03
        self._milestone_still_speed = 0.18

        self.boundary_x = float(task_cfg.boundary_x)
        self.boundary_y_max = float(task_cfg.boundary_y_max)
        self.tilt_threshold_deg = float(task_cfg.tilt_threshold_deg)
        self.gravity_z_termination_threshold = -np.cos(np.deg2rad(self.tilt_threshold_deg))

        # 里程碑庆祝动作控制：先静止1.5秒，再保持预设姿势1.5秒
        self._celebration_hold_seconds = 1.5  # 静止阶段
        self._celebration_pose_seconds = 1.5  # 姿势保持阶段
        steps_per_second = cfg.max_episode_steps / max(cfg.max_episode_seconds, 1e-6)
        self._celebration_hold_steps = max(1, int(self._celebration_hold_seconds * steps_per_second))
        self._celebration_pose_steps = max(1, int(self._celebration_pose_seconds * steps_per_second))
        
        # 庆祝预设姿势：前腿抬起+后腿下蹲（类似宇树Go1跪姿抬手）
        # 关节顺序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        #           RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
        self._celebration_pose = np.array([
            0.0,  -0.3, -0.5,   # FR: 抬起（大腿前伸，小腿微收）
            0.0,  -0.3, -0.5,   # FL: 抬起
            0.0,   1.5, -2.5,   # RR: 下蹲（大腿收紧，小腿深弯）
            0.0,   1.5, -2.5,   # RL: 下蹲
        ], dtype=np.float32)
    
        # ========== 硬编码导航调试模式 ==========
        # 设为True可让机器人按预定顺序访问所有地标点，用于测试最优路径和奖励上界
        self.use_hardcoded_navigation = True  # 改为True时启用
        self._build_hardcoded_waypoints()
        self._hardcoded_milestone_wp_idx = np.array([6, 14, 23], dtype=np.int32)
    
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
        """构建完整全局硬编码路径（24个waypoints, 索引0-23）
        阶段1：START(笑脸左) → 笑脸中→右 → 红包×3 → 2026平台（庆祝）
        阶段2：楼梯口 → 上楼梯 → 吊桥拜年红包 → 下楼梯 → 丙午大吉（庆祝）
        阶段3：河床红包×6（含拜年） → 滚球区 → 不规则地形 → 终点（庆祝）
        
        里程碑平台位置: [wp6, wp14, wp23] = [2026平台, 丙午大吉, 终点]
        红包总数：3（section011）+ 1（吊桥）+ 6（河床含拜年） = 10个
        """
        waypoints = []
        
        # === 阶段1：Section011（坑洼地形）===
        waypoints.append([-3.0, 0.1])  # START (修改为笑脸左位置，避免出生随机波动还要导航回起点)
        
        # 笑脸×3：左→中→右 (waypoints[1-3]，第一个笑脸与起点重合)
        waypoints.extend([[0.0, 0.1], [3.0, 0.1]])
        
        # 红包×3：右→中→左（之字形）
        waypoints.extend([[3.0, 4.1], [0.0, 4.1], [-3.0, 4.1]])
        
        # 2026平台（第1个里程碑）
        waypoints.append([-3.0, 8.0])
        
        # === 阶段2：Section012（楼梯+吊桥）===
        waypoints.append([-3.0, 12.0])  # 左侧楼梯口
        waypoints.append([-3.0, 13.0])  # 上楼梯Step1
        waypoints.append([-3.0, 14.0])  # 上楼梯Step2
        waypoints.append([-3.0, 15.0])  # 上楼梯Step3
        waypoints.append([-3.0, 16.5])  # 上楼梯Step4
        waypoints.append([-3.0, 18.1])  # 吊桥拜年红包
        waypoints.append([-3.0, 21.0])  # 准备下楼梯
        
        # 丙午大吉平台（第2个里程碑）
        waypoints.append([0.0, 24.0])
        
        # === 阶段3：Section013（河床×6+滚球+不规则地形）===
        # 河床收集6个红包（5个+1个拜年）
        waypoints.append([0.5, 19.5])   # 河床红包1
        waypoints.append([-2.9, 18.1])  # 河床拜年红包
        waypoints.append([0.5, 16.0])   # 河床红包2
        waypoints.append([2.0, 18.0])   # 河床红包3
        waypoints.append([3.5, 16.0])   # 河床红包4
        waypoints.append([3.5, 19.5])   # 河床红包5（不是出口，就是红包）
        
        # 下楼梯经过平台到滚球区域
        waypoints.append([0.0, 25.5])   # 滚动球区域
        waypoints.append([3.0, 26.2])   # 横向移动确保碰球
        
        # 最终终点（第3个里程碑）
        waypoints.append([0.0, 32.0])
        
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
        """更新目标位置标记的位置和朝向（向量化）"""
        all_dof_pos = data.dof_pos.copy()
        # 批量赋值：pose_commands[:, :3] = [x, y, yaw]
        all_dof_pos[:, self._target_marker_dof_start:self._target_marker_dof_end] = pose_commands[:, :3]
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（向量化，不逐环境循环）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5
        all_dof_pos = data.dof_pos.copy()
        arrow_heights = robot_pos[:, 2] + arrow_offset  # [num_envs]
        
        # --- 当前运动方向箭头（批量） ---
        cur_speed = np.linalg.norm(base_lin_vel_xy, axis=1)  # [num_envs]
        cur_yaw = np.where(
            cur_speed > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0
        )  # [num_envs]
        robot_arrow_quats = self._euler_to_quat_batch(np.zeros(num_envs), np.zeros(num_envs), cur_yaw)  # [num_envs, 4]
        # 归一化
        quat_norms = np.linalg.norm(robot_arrow_quats, axis=1, keepdims=True)
        robot_arrow_quats = np.where(quat_norms > 1e-6, robot_arrow_quats / quat_norms, np.array([[0.0, 0.0, 0.0, 1.0]]))
        # 拼接 pos+quat
        robot_arrow_pos = np.column_stack([robot_pos[:, 0], robot_pos[:, 1], arrow_heights])  # [num_envs, 3]
        all_dof_pos[:, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([robot_arrow_pos, robot_arrow_quats], axis=1)
        
        # --- 期望运动方向箭头（批量） ---
        des_speed = np.linalg.norm(desired_vel_xy, axis=1)
        des_yaw = np.where(
            des_speed > 1e-3,
            np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]),
            0.0
        )
        desired_arrow_quats = self._euler_to_quat_batch(np.zeros(num_envs), np.zeros(num_envs), des_yaw)
        quat_norms = np.linalg.norm(desired_arrow_quats, axis=1, keepdims=True)
        desired_arrow_quats = np.where(quat_norms > 1e-6, desired_arrow_quats / quat_norms, np.array([[0.0, 0.0, 0.0, 1.0]]))
        desired_arrow_pos = np.column_stack([robot_pos[:, 0], robot_pos[:, 1], arrow_heights])
        all_dof_pos[:, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([desired_arrow_pos, desired_arrow_quats], axis=1)
        
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
    
    def _euler_to_quat_batch(self, roll, pitch, yaw):
        """批量欧拉角转四元数 [num_envs, 4] (qx,qy,qz,qw) - 向量化版本"""
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
        
        return np.column_stack([qx, qy, qz, qw]).astype(np.float32)
    
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

            # 关键优化：里程碑路点必须先完成庆祝动作，才允许切到下一个路点
            wp_idx = state.info["hardcoded_waypoint_idx"]
            block_advance = np.zeros(data.shape[0], dtype=bool)
            milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
            for m_idx, m_name in zip(self._hardcoded_milestone_wp_idx, milestone_names):
                done_key = f"{m_name}_celebration_done"
                celebration_done = state.info.get(done_key, np.zeros(data.shape[0], dtype=bool))
                block_advance |= (wp_idx == m_idx) & (~celebration_done)

            reached_wp_for_advance = reached_wp & (~block_advance)
            
            # 更新到下一路点（不超过最后一个）
            max_wp_idx = len(self._hardcoded_waypoints) - 1
            state.info["hardcoded_waypoint_idx"] = np.minimum(
                state.info["hardcoded_waypoint_idx"] + reached_wp_for_advance.astype(np.int32),
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

        # ========== 里程碑庆祝动作编排（静止1.5秒 -> 预设姿势1.5秒） ==========
        # 注：仅影响速度命令，不改变红包判定逻辑
        milestone_ys = self._milestone_positions[:, 1].tolist()
        milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
        robot_y = root_pos[:, 1]

        for milestone_y, milestone_name in zip(milestone_ys, milestone_names):
            done_key = f"{milestone_name}_celebration_done"
            hold_key = f"{milestone_name}_hold_steps"
            pose_key = f"{milestone_name}_pose_steps"

            if done_key not in state.info:
                state.info[done_key] = np.zeros(data.shape[0], dtype=bool)
            if hold_key not in state.info:
                state.info[hold_key] = np.zeros(data.shape[0], dtype=np.int32)
            if pose_key not in state.info:
                state.info[pose_key] = np.zeros(data.shape[0], dtype=np.int32)

            at_milestone = (robot_y >= milestone_y - 0.5) & (robot_y <= milestone_y + 2.0)
            celebration_active = at_milestone & (~state.info[done_key])

            # 第一阶段：静止1.5秒
            in_hold = celebration_active & (state.info[hold_key] < self._celebration_hold_steps)
            state.info[hold_key] = np.where(
                in_hold,
                state.info[hold_key] + 1,
                np.where(~celebration_active, 0, state.info[hold_key]),
            )

            # 第二阶段：保持预设姿势1.5秒
            in_pose = celebration_active & (state.info[hold_key] >= self._celebration_hold_steps)
            state.info[pose_key] = np.where(
                in_pose,
                state.info[pose_key] + 1,
                np.where(~celebration_active, 0, state.info[pose_key]),
            )

            # 静止阶段：保持默认姿态（速度和角速度都为0）
            desired_vel_xy[in_hold, :] = 0.0
            desired_yaw_rate[in_hold] = 0.0

            # 姿势阶段：也保持静止（姿势通过奖励引导，不是速度命令）
            desired_vel_xy[in_pose, :] = 0.0
            desired_yaw_rate[in_pose] = 0.0

            # 接近平台时减速（防止冲出平台）
            approaching_milestone = at_milestone & (~celebration_active)
            if np.any(approaching_milestone):
                speed_limit = np.linalg.norm(desired_vel_xy[approaching_milestone, :], axis=1, keepdims=True)
                desired_vel_xy[approaching_milestone, :] = (
                    desired_vel_xy[approaching_milestone, :] / np.maximum(speed_limit, 1e-6) * 0.25
                )
                desired_yaw_rate[approaching_milestone] = desired_yaw_rate[approaching_milestone] * 0.3

            # 庆祝完成判定：姿势阶段保持够1.5秒
            pose_complete = in_pose & (state.info[pose_key] >= self._celebration_pose_steps)
            state.info[done_key] = state.info[done_key] | pose_complete
        
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
        info = terminated_state.info
        
        # ========== 关键指标统计（用于训练诊断） ==========
        self.navigation_stats_step += 1
        robot_y = root_pos[:, 1]
        avg_y = np.mean(robot_y)
        avg_reward = np.mean(reward)
        
        # 最远waypoint统计
        current_wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(data.shape[0], dtype=np.int32))
        max_wp_reached = np.max(current_wp_idx)
        
        # 楼梯通过率（Y > 21）
        stair_pass_ratio = np.mean(robot_y > 21.0)
        
        # 卡住比例
        stuck_ratio = np.mean(info.get("stuck_counter", np.zeros(data.shape[0], dtype=np.int32)) > 100)
        
        # 添加到窗口
        if len(self.metric_window['robot_y']) < self.metric_window_size:
            self.metric_window['robot_y'].append(avg_y)
            self.metric_window['rewards'].append(avg_reward)
            self.metric_window['stair_pass'].append(stair_pass_ratio)
            self.metric_window['stuck_ratio'].append(stuck_ratio)
            self.metric_window['max_waypoint'].append(max_wp_reached)
        else:
            # 滑动窗口
            self.metric_window['robot_y'] = self.metric_window['robot_y'][1:] + [avg_y]
            self.metric_window['rewards'] = self.metric_window['rewards'][1:] + [avg_reward]
            self.metric_window['stair_pass'] = self.metric_window['stair_pass'][1:] + [stair_pass_ratio]
            self.metric_window['stuck_ratio'] = self.metric_window['stuck_ratio'][1:] + [stuck_ratio]
            self.metric_window['max_waypoint'] = self.metric_window['max_waypoint'][1:] + [max_wp_reached]
        
        # 每1000步打印一次诊断信息
        if self.navigation_stats_step % 1000 == 0:
            self._print_training_metrics()
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        state.info = info
        
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
    
    def _print_training_metrics(self):
        """打印训练诊断指标"""
        import sys
        
        if len(self.metric_window['robot_y']) == 0:
            return
        
        avg_y = np.mean(self.metric_window['robot_y'])
        avg_reward = np.mean(self.metric_window['rewards'])
        avg_stair = np.mean(self.metric_window['stair_pass'])
        avg_stuck = np.mean(self.metric_window['stuck_ratio'])
        max_wp = int(np.max(self.metric_window['max_waypoint']))
        
        # 计算改进速度
        y_improve = 0
        reward_improve = 0
        if len(self.metric_window['robot_y']) > 500:
            y_improve = (np.mean(self.metric_window['robot_y'][-250:]) - 
                         np.mean(self.metric_window['robot_y'][:250])) / 250
            reward_improve = (np.mean(self.metric_window['rewards'][-250:]) - 
                             np.mean(self.metric_window['rewards'][:250])) / 250
        
        # 格式化输出
        print("\n" + "="*90, file=sys.stderr)
        print(f"📊【训练进度】步数: {self.navigation_stats_step:,}", file=sys.stderr)
        print("="*90, file=sys.stderr)
        
        print(f"【航点】最远达到: wp{max_wp}/24 ({max_wp/24*100:.1f}%)", file=sys.stderr)
        if max_wp >= 24:
            print("       ✅ 完成全程", file=sys.stderr)
        elif max_wp >= 15:
            print("       ✅ 已过楼梯", file=sys.stderr)
        elif max_wp >= 8:
            print("       ⚠️  进入楼梯", file=sys.stderr)
        elif max_wp >= 7:
            print("       ⚠️  到达2026平台", file=sys.stderr)
        else:
            print("       ❌ 尚在坑洼区", file=sys.stderr)
        
        print(f"【进度】avg_y={avg_y:.2f}m  增速={y_improve:.4f}m/step", file=sys.stderr)
        if y_improve > 0.002:
            print("       ✅ 快速进度", file=sys.stderr)
        elif y_improve > 0.0005:
            print("       ⚠️  缓慢进度", file=sys.stderr)
        else:
            print("       ❌ 停滞", file=sys.stderr)
        
        print(f"【奖励】avg={avg_reward:.2f}分  改进={reward_improve:.4f}/step", file=sys.stderr)
        if avg_reward > 50:
            print("       ✅ 学习快速", file=sys.stderr)
        elif avg_reward > 10:
            print("       ⚠️  学习正常", file=sys.stderr)
        else:
            print("       ❌ 无学习信号", file=sys.stderr)
        
        print(f"【楼梯】通过率(Y>21)={avg_stair*100:.1f}%", file=sys.stderr)
        if avg_stair > 0.8:
            print("       ✅ 完全掌握", file=sys.stderr)
        elif avg_stair > 0.5:
            print("       ⚠️  基本通过", file=sys.stderr)
        elif avg_stair > 0.2:
            print("       ⚠️  部分通过", file=sys.stderr)
        else:
            print("       ❌ 卡住", file=sys.stderr)
        
        print(f"【卡住】卡住比例={avg_stuck*100:.1f}%", file=sys.stderr)
        if avg_stuck < 0.1:
            print("       ✅ 很少卡住", file=sys.stderr)
        elif avg_stuck < 0.3:
            print("       ⚠️  有时卡住", file=sys.stderr)
        else:
            print("       ❌ 经常卡住", file=sys.stderr)
        
        print("="*90 + "\n", file=sys.stderr)
    
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
        
        # 楼梯段特殊奖励倍数（Y=12-21最陡峭）
        in_stairs = (robot_y >= 12.0) & (robot_y <= 21.0)
        stair_bonus = np.where(in_stairs, 1.8, 1.2)  # 楼梯段奖励1.8x，其他段1.2x
        reward += stair_bonus * forward_progress  # 在楼梯段给更强的前进奖励
        
        # 3. 弱朝向对齐（帮助初始转向：机器人朝+Y时目标也在+Y，很容易对齐）
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(goal_dir[:, 1], goal_dir[:, 0]).flatten()
        heading_error = desired_heading - robot_heading
        heading_error = np.where(heading_error > np.pi, heading_error - 2*np.pi, heading_error)
        heading_error = np.where(heading_error < -np.pi, heading_error + 2*np.pi, heading_error)
        heading_align = np.exp(-np.square(heading_error / 0.6))
        reward += 0.2 * heading_align  # 降低到0.2，不压制横向收集地标
        
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
        
        # 楼梯段更严格的稳定性要求（楼梯摔倒后果更严重）
        soft_tilt_penalty = np.where(
            tilt_cos > 0.707,  # cos(45°)=0.707, tilt<45°: 无惩罚
            0.0,
            -0.002 * (0.707 - tilt_cos) / 0.707  # 最大约-0.002/步
        )
        # 在楼梯段加强稳定性惩罚（倾斜太大就重惩罚）
        stair_stability_penalty = np.where(
            in_stairs & (tilt_cos < 0.6),  # 楼梯上倾斜>50°就重罚
            -0.1 * (0.6 - tilt_cos),
            0.0
        )
        reward += soft_tilt_penalty + stair_stability_penalty
        
        # 3. 卡住时鼓励旋转探索（v7.7: 放松旋转判定，更积极鼓励遇到难点旋转通过）
        # 当stuck_counter>80步(0.8秒)时，奖励角速度，鼓励转向寻找新路径
        stuck_counter = info.get("stuck_counter", np.zeros(num_envs, dtype=np.int32))
        is_stuck = stuck_counter > 80  # 降低阈值：100→80步，更早鼓励旋转
        angular_speed = np.abs(gyro[:, 2])  # 角速度绝对值（转得越快越好）
        rotation_explore_reward = np.where(
            is_stuck,
            np.clip(angular_speed * 1.2, 0, 0.6),  # 增强旋转奖励：0.8→1.2，上限0.4→0.6
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
        
        # [已删除] 地标收集奖励与收集率优化（改用硬编码waypoints导航）
        
        # ========== 4. 多里程碑平台奖励（3个平台） ==========
        # 里程碑: Y=8(2026平台), Y=24(丙午大吉), Y=32(终点)
        milestone_ys = self._milestone_positions[:, 1].tolist()
        milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
        
        for milestone_y, milestone_name in zip(milestone_ys, milestone_names):
            milestone_key = f"{milestone_name}_reached"
            if milestone_key not in info:
                info[milestone_key] = np.zeros(num_envs, dtype=bool)
            
            # 到达里程碑判断（Y坐标超过且在±2m范围内）
            reached_milestone = (robot_y >= milestone_y - 0.5) & (robot_y <= milestone_y + 2.0)
            first_reach_milestone = reached_milestone & (~info[milestone_key])
            
            # 里程碑奖励（固定值，不再受收集率影响）
            milestone_bonus = 50.0
            
            reward += first_reach_milestone.astype(np.float32) * milestone_bonus
            info[milestone_key] = info[milestone_key] | reached_milestone

        # 平台驻留奖励：鼓励在里程碑中心减速停稳，提升庆祝动作触发概率
        planar_speed = np.linalg.norm(root_vel[:, :2], axis=1)
        for i, (milestone_y, milestone_name) in enumerate(zip(milestone_ys, milestone_names)):
            center_xy = self._milestone_positions[i, :2]
            dist_to_center = np.linalg.norm(robot_xy - center_xy, axis=1)
            at_center = dist_to_center < self._milestone_stay_radius
            at_milestone = (robot_y >= milestone_y - 0.5) & (robot_y <= milestone_y + 2.0)
            done_key = f"{milestone_name}_celebration_done"
            celebration_done = info.get(done_key, np.zeros(num_envs, dtype=bool))

            stay_reward = at_milestone & at_center & (~celebration_done)
            still_reward = stay_reward & (planar_speed < self._milestone_still_speed)

            reward += stay_reward.astype(np.float32) * self._milestone_hold_reward
            reward += still_reward.astype(np.float32) * self._milestone_still_reward
        
        # 最终终点特殊奖励
        reached_final_goal = robot_y >= 32.0
        goal_reached_key = "goal_reached"
        if goal_reached_key not in info:
            info[goal_reached_key] = np.zeros(num_envs, dtype=bool)
        
        first_reach_final = reached_final_goal & (~info[goal_reached_key])
        final_bonus = np.full(num_envs, self.goal_reached_reward, dtype=np.float32)
        
        reward += first_reach_final.astype(np.float32) * final_bonus
        info[goal_reached_key] = info[goal_reached_key] | reached_final_goal
        
        # ========== 5. 多平台庆祝动作检测（预设姿势） ==========
        # 在每个里程碑平台执行预设的“抬前腿蹲坐”姿势
        for milestone_y, milestone_name in zip(milestone_ys, milestone_names):
            celebration_done_key = f"{milestone_name}_celebration_done"
            pose_key = f"{milestone_name}_pose_steps"
            
            if celebration_done_key not in info:
                info[celebration_done_key] = np.zeros(num_envs, dtype=bool)
            if pose_key not in info:
                info[pose_key] = np.zeros(num_envs, dtype=np.int32)
            
            # 是否在当前里程碑平台
            at_milestone = (robot_y >= milestone_y - 0.5) & (robot_y <= milestone_y + 2.0)
            
            # 姿势阶段奖励：关节角度接近目标预设越近奖励越大
            in_pose = at_milestone & (~info[celebration_done_key]) & (info.get(pose_key, np.zeros(num_envs, dtype=np.int32)) > 0)
            if np.any(in_pose):
                # 计算当前关节角度与目标姿势的距离
                joint_error = np.linalg.norm(joint_pos[in_pose] - self._celebration_pose, axis=1)
                # 奖励接近目标姿势（误差越小奖励越大）
                pose_reward = np.exp(-joint_error * 2.0) * 0.1  # 每步最大≈+0.1
                reward[in_pose] += pose_reward
            
            # 庆祝完成奖励：姿势保持够时间就给大奖励
            if self.enable_celebration_reward:
                celebration_complete = (
                    at_milestone
                    & (~info[celebration_done_key])
                    & (info.get(pose_key, np.zeros(num_envs, dtype=np.int32)) >= self._celebration_pose_steps)
                )
                reward += celebration_complete.astype(np.float32) * self.celebration_reward
                info[celebration_done_key] = info[celebration_done_key] | celebration_complete
        
        # ========== 6. 稳定性惩罚 ==========
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
        # 让机器人既学习完整全程的地形特征，又有多样化的起始位置
        # 50%环境：从赛道中间随机位置出生（训练多样性）
        # 50%环境：从起点出生（训练完整全程）
        num_curriculum = int(num_envs * self._curriculum_spawn_fraction)
        if num_curriculum > 0:
            cur_idx = np.arange(num_curriculum)  # 前70%环境用课程出生
            # Y: 从起点到更远处随机，覆盖楼梯下后的区域（扩大学习范围）
            cur_y = np.random.uniform(
                self.spawn_center[1], self.goal_y + 5.0,  # 扩大到goal_y+5m，覆盖section013前段
                size=num_curriculum
            )
            # X: 赛道宽度内随机（避开墙壁）
            cur_x = np.random.uniform(-4.0, 4.0, size=num_curriculum)
            robot_init_xy[cur_idx, 0] = cur_x
            robot_init_xy[cur_idx, 1] = cur_y
            terrain_heights[cur_idx] = self._curriculum_spawn_z  # 高处出生（2.0m），确保在楼梯/河床/吊桥等复杂地形也能安全着陆
        
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
        
        # ========== 初始朝向随机化（比赛条件）==========
        # 机器人应朝向+Y方向（目标方向），±20°随机化
        init_yaw = np.pi / 2.0  # 90° 对应+Y方向
        yaw_jitter = np.deg2rad(20.0)  # ±20°随机化
        random_yaw = np.random.uniform(init_yaw - yaw_jitter, init_yaw + yaw_jitter, size=num_envs)
        
        # 批量计算base四元数（DOF 6-9）
        base_quats = self._euler_to_quat_batch(np.zeros(num_envs), np.zeros(num_envs), random_yaw)
        dof_pos[:, self._base_quat_start:self._base_quat_end] = base_quats
        
        # 批量设置箭头四元数（如果箭头body存在）
        if self._robot_arrow_body is not None:
            # 机器人方向箭头 = 与机器人朝向一致
            robot_arrow_quats = self._euler_to_quat_batch(np.zeros(num_envs), np.zeros(num_envs), random_yaw)
            dof_pos[:, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quats
            
            # 期望方向箭头 = 初始指向+Y方向
            desired_yaw = np.full(num_envs, np.pi / 2.0)
            desired_arrow_quats = self._euler_to_quat_batch(np.zeros(num_envs), np.zeros(num_envs), desired_yaw)
            dof_pos[:, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quats
        
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
    