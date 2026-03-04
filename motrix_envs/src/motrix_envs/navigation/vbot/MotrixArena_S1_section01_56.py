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
            'descent_pass': [],  # v7.9: 下楼梯完成率（Y>24，从wp13到达丙午平台）
            'stuck_ratio': [],
            'max_waypoint': []
        }
        self.metric_window_size = 1000

        # ========== 课程学习v3（Frontier-Biased Curriculum） ==========
        # 核心思路：跟踪训练前沿(global_max_wp)，自动把训练资源集中在"还没学会的地方"
        # 30%环境：从起点出发（保持完整路线能力，防遗忘）
        # 70%环境：课程出生，其中60%在前沿附近，40%在已学区域复习
        self._curriculum_spawn_fraction = 0.7  # 70%环境用课程出生
        self._curriculum_spawn_z = 2.0  # 课程出生高度（2.0m），足够高避免卡地形
        self._global_max_wp = 0  # 训练中所有env曾到达的最远waypoint（动态更新）

        # ========== 从cfg读取阶段任务参数 ==========
        task_cfg = cfg.task_config
        self.task_name = task_cfg.task_name
        self.enable_celebration_reward = task_cfg.enable_celebration_reward

        self.goal_y = float(task_cfg.goal_y)
        self.goal_reached_reward = float(task_cfg.goal_reached_reward)

        self.celebration_reward = float(task_cfg.celebration_reward)
        self._milestone_positions = np.array(task_cfg.milestone_positions, dtype=np.float32)
        if self._milestone_positions.size == 0:
            self._milestone_positions = np.array([[-3.0, 7.7], [0.0, 24.2], [0.0, 32.0]], dtype=np.float32)
        self._milestone_y_tolerance = 0.5   # v7.18: 平台到达窗口：Y ±0.5（官方平台1m宽）
        self._milestone_x_abs_limit = 3.0   # 平台到达窗口：|X| <= 3.0
        self._milestone_stay_radius = 1.0
        self._milestone_hold_reward = 0.2
        self._milestone_still_reward = 0.3
        self._milestone_still_speed = 0.1

        self.boundary_x = float(task_cfg.boundary_x)
        self.boundary_y_max = float(task_cfg.boundary_y_max)
        self.tilt_threshold_deg = float(task_cfg.tilt_threshold_deg)
        self.gravity_z_termination_threshold = -np.cos(np.deg2rad(self.tilt_threshold_deg))

        # 里程碑庆祝动作控制：先静止1秒，再保持预设姿势1秒，恢复2秒
        self._celebration_hold_seconds = 1.5  # v7.18: 静止阶段 1.5
        self._celebration_pose_seconds = 1.0  # v7.18: 姿势保持 1.5→1.0
        steps_per_second = cfg.max_episode_steps / max(cfg.max_episode_seconds, 1e-6)
        self._celebration_hold_steps = max(1, int(self._celebration_hold_seconds * steps_per_second))
        self._celebration_pose_steps = max(1, int(self._celebration_pose_seconds * steps_per_second))
        self._celebration_exit_steps = max(1, int(2.2 * steps_per_second))  # v7.18: 恢复时间 1.5→2.2秒
        
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
        self._hardcoded_milestone_wp_idx = np.array([6, 16, 31], dtype=np.int32)  # v7.18: 新增wp后终点索引28->30，本次新增wp后->31
    
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
        """构建完整全局硬编码路径（32个waypoints, 索引0-31）
        阶段1：START(笑脸左) → 笑脸中→右 → 红包×3 → 2026平台（庆祝）
        阶段2：楼梯口 → 上楼梯 → 吊桥拜年红包 → 下楼梯 → 丙午大吉（庆祝）
        阶段3：河床红包×6（含拜年） → 滚球区 → 不规则地形 → 终点（庆祝）
        
        里程碑平台位置: [wp6, wp16, wp31] = [2026平台, 丙午大吉, 终点]
        
        v7.18变更：
        - wp19: [0,20]→[0,21]，wp20新增[0.5,21]
        - wp25新增[2.7,15]（河床红包3→4过渡）
        - 总数29→31，终点wp28→wp30
        
        本次变更：
        - wp28新增[2.0,23.5]（河床红包5→滚球区过渡）
        - 总数31→32，终点wp30→wp31
        """
        waypoints = []
        
        # === 阶段1：Section011（坑洼地形）===
        waypoints.append([-3.0, 0.1])   # wp0: START
        waypoints.extend([[0.0, 0.1], [3.0, 0.1]])  # wp1-2: 笑脸中→右
        waypoints.extend([[3.0, 4.1], [0.0, 4.1], [-3.0, 4.1]])  # wp3-5: 红包右→中→左（wp5开始减速到2026）
        waypoints.append([-3.0, 7.7])   # wp6: 2026平台（里程碑1，官方Y=7.2~8.2）
        
        # === 阶段2：Section012（楼梯+吊桥）===
        waypoints.append([-3.0, 12.0])  # wp7: 左侧楼梯口
        waypoints.append([-3.0, 13.0])  # wp8: 上楼梯Step1
        waypoints.append([-3.0, 14.0])  # wp9: 上楼梯Step2
        waypoints.append([-3.0, 15.0])  # wp10: 上楼梯Step3
        waypoints.append([-3.0, 16.5])  # wp11: 上楼梯Step4
        waypoints.append([-3.0, 18.1])  # wp12: 吊桥拜年红包（之后开始半速）
        waypoints.append([-3.0, 21.0])  # wp13: 准备下楼梯
        waypoints.append([-3.0, 22.0])  # wp14: 下楼梯过渡
        waypoints.append([-3.0, 23.0])  # wp15: 丙午平台入口过渡
        waypoints.append([0.0, 24.2])   # wp16: 丙午大吉平台（里程碑2，官方Y=23.7~24.7）

        # 丙午后反向上楼梯过渡（更稳地回到河床）
        waypoints.append([0.1, 23.0])   # wp17: 反向楼梯1
        waypoints.append([0.3, 22.0])   # wp18: 反向楼梯2
        waypoints.append([0.4, 21.0])   # wp19: 反向楼梯3（v7.18: 20→21）
        waypoints.append([0.5, 21.0])   # wp20: 反向楼梯出口（v7.18新增）
        
        # === 阶段3：Section013（河床×6+滚球+不规则地形）===
        # 半速区域：wp12(吊桥)之后 ~ wp28(滚球)之前
        waypoints.append([0.5, 19.5])   # wp21: 河床红包1
        waypoints.append([-2.9, 18.1])  # wp22: 河床拜年红包
        waypoints.append([0.5, 16.0])   # wp23: 河床红包2
        waypoints.append([2.0, 18.0])   # wp24: 河床红包3
        waypoints.append([2.7, 14.0])   # wp25: 河床过渡（v7.18新增）
        waypoints.append([3.5, 16.0])   # wp26: 河床红包4
        waypoints.append([3.5, 19.5])   # wp27: 河床红包5
        waypoints.append([2.0, 23.5])   # wp28: 新增过渡点
        
        # 结束半速区域，恢复正常速度
        waypoints.append([0.0, 25.5])   # wp29: 滚动球区域
        waypoints.append([3.0, 26.2])   # wp30: 横向移动确保碰球
        waypoints.append([0.0, 32.0])   # wp31: 最终终点（里程碑3）
        
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

        # 计算默认力矩
        torques = self._compute_torques(state.info["filtered_actions"], state.data)
        
        # === 庆祝动作覆盖：在庆祝阶段直接PD控制，不依赖策略网络 ===
        torques = self._override_celebration_torques(torques, state)
        
        state.data.actuator_ctrls = torques
        return state
    
    def _override_celebration_torques(self, torques: np.ndarray, state: NpEnvState) -> np.ndarray:
        """庆祝动作力矩覆盖：在庆祝阶段直接PD控制关节到预设姿势
        
        核心思路：不依赖RL策略自己发现12维目标姿势（高维空间极难收敛），
        而是在检测到庆祝阶段时，直接控制关节到预设姿势，确保庆祝动作100%执行。
        RL只需学会"走到平台并停下来"，系统自动接管庆祝姿势的执行。
        
        阶段1（静止Hold）：PD控制到默认站姿，让机器人平稳停下
        阶段2（姿势Pose）：PD控制到庆祝预设姿势（前腿抬起+后腿下蹲）
        """
        milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
        kp, kv = 80.0, 8.0  # 庆祝用稍高阻尼，确保平稳过渡
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        
        for milestone_name in milestone_names:
            hold_key = f"{milestone_name}_hold_steps"
            done_key = f"{milestone_name}_celebration_done"
            exit_key = f"{milestone_name}_exit_steps"
            
            hold_steps = state.info.get(hold_key, None)
            done = state.info.get(done_key, None)
            if exit_key not in state.info:
                state.info[exit_key] = np.zeros(state.data.shape[0], dtype=np.int32)
            
            if hold_steps is None or done is None:
                continue
            
            # 静止阶段：PD控制到默认站姿（让机器人平稳停下）
            in_hold = (~done) & (hold_steps > 0) & (hold_steps < self._celebration_hold_steps)
            if np.any(in_hold):
                cur_pos = self.get_dof_pos(state.data)[in_hold]
                cur_vel = self.get_dof_vel(state.data)[in_hold]
                target = np.tile(self.default_angles, (np.sum(in_hold), 1))
                hold_torques = kp * (target - cur_pos) - kv * cur_vel
                torques[in_hold] = np.clip(hold_torques, -torque_limits, torque_limits)
            
            # 姿势阶段：PD控制到庆祝预设姿势（前腿抬起+后腿下蹲）
            in_pose = (~done) & (hold_steps >= self._celebration_hold_steps)
            if np.any(in_pose):
                cur_pos = self.get_dof_pos(state.data)[in_pose]
                cur_vel = self.get_dof_vel(state.data)[in_pose]
                target = np.tile(self._celebration_pose, (np.sum(in_pose), 1))
                pose_torques = kp * (target - cur_pos) - kv * cur_vel
                torques[in_pose] = np.clip(pose_torques, -torque_limits, torque_limits)

            # 退出阶段：庆祝完成后平滑回默认站姿，避免突然起身蹦起
            in_exit = done & (state.info[exit_key] < self._celebration_exit_steps)
            if np.any(in_exit):
                cur_pos = self.get_dof_pos(state.data)[in_exit]
                cur_vel = self.get_dof_vel(state.data)[in_exit]
                alpha = (state.info[exit_key][in_exit] / max(1, self._celebration_exit_steps)).astype(np.float32)[:, None]
                target = (1.0 - alpha) * self._celebration_pose[None, :] + alpha * self.default_angles[None, :]
                exit_torques = kp * (target - cur_pos) - kv * cur_vel
                torques[in_exit] = np.clip(exit_torques, -torque_limits, torque_limits)

            state.info[exit_key] = np.where(done, np.minimum(state.info[exit_key] + 1, self._celebration_exit_steps), 0)
        
        return torques
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 85.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel

        # v7.15: 地形相关力矩缩放
        # - 上楼梯时增强后腿推力，减少“前腿使劲后腿悬空”
        # - 下楼梯时整体略降力矩，避免冲得过快
        root_pos, _, _ = self._extract_root_state(data)
        # 区域判断全部用wp_idx
        wp_idx = getattr(self, '_cached_wp_idx', np.zeros(data.shape[0], dtype=np.int32))
        # 上楼梯区：wp7~wp11
        # 下楼梯区：wp13~wp15
        in_stairs_up = (wp_idx >= 8) & (wp_idx <= 10)
        in_stairs_down = (wp_idx >= 13) & (wp_idx <= 15)

        if np.any(in_stairs_up):
            # 后腿: RR(6,7,8), RL(9,10,11)
            torques[np.ix_(in_stairs_up, [6, 7, 8, 9, 10, 11])] *= 1.15
        if np.any(in_stairs_down):
            torques[in_stairs_down] *= 0.9
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_target_speed(self, wp_idx):
        """计算地形自适应目标速度"""
        # 波浪/坑洼地形：wp0~wp2
        # 2026平台前减速区：wp5~wp6
        # 上楼梯区：wp7~wp11
        # 下楼梯/吊桥区：wp13~wp15
        in_wave_terrain = (wp_idx >= 0) & (wp_idx <= 2)
        in_approach_2026 = (wp_idx >= 5) & (wp_idx <= 6)
        in_stairs_up = (wp_idx >= 8) & (wp_idx <= 10)
        in_bridge = (wp_idx >= 10) & (wp_idx <= 13)
        in_stairs_down = (wp_idx >= 13) & (wp_idx <= 15)
        in_reverse_stairs_up = (wp_idx >= 17) & (wp_idx <= 19)
        in_riverbed = (wp_idx >= 20) & (wp_idx <= 27)

        return np.where(
            in_stairs_down, 0.10,
            np.where(in_stairs_up, 0.5,
            np.where(in_reverse_stairs_up, 0.5,
            np.where(in_riverbed, 0.6,
            np.where(in_bridge, 0.10,
            np.where(in_approach_2026, 0.18,
            np.where(in_wave_terrain, 1.1, 1.0)))))))

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
    
    def _compute_roll_pitch_from_quat(self, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从四元数计算Roll和Pitch角（弧度），用于IMU侧翻检测
        
        Roll:  绕X轴旋转（左右倾斜）
        Pitch: 绕Y轴旋转（前后俯仰）
        
        Returns:
            roll:  [num_envs] 弧度，正值=向右倾
            pitch: [num_envs] 弧度，正值=向前俯
        """
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation) - clamp for numerical stability near gimbal lock
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        return roll, pitch
    
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

    def _in_milestone_zone(self, robot_x: np.ndarray, robot_y: np.ndarray, milestone_y: float) -> np.ndarray:
        """里程碑检测窗口：|x|<=3 且 |y-milestone_y|<=0.4"""
        in_y = np.abs(robot_y - milestone_y) <= self._milestone_y_tolerance
        in_x = np.abs(robot_x) <= self._milestone_x_abs_limit
        return in_x & in_y
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # v7.17: 出生步数计数（用于保护期和庆祝延迟）
        if "spawn_steps" not in state.info:
            state.info["spawn_steps"] = np.zeros(data.shape[0], dtype=np.int32)
        state.info["spawn_steps"] += 1

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
            
            # 检测是否到达当前路点（距离<0.3m，更严格）
            dist_to_wp = np.linalg.norm(robot_xy - current_waypoints, axis=1)
            reached_wp = dist_to_wp < 0.3

            # 优化2026平台（wp6）庆祝逻辑
            wp_idx = state.info["hardcoded_waypoint_idx"]
            block_advance = np.zeros(data.shape[0], dtype=bool)
            milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
            for m_idx, m_name in zip(self._hardcoded_milestone_wp_idx, milestone_names):
                done_key = f"{m_name}_celebration_done"
                celebration_done = state.info.get(done_key, np.zeros(data.shape[0], dtype=bool))
                # 2026平台（wp6）和丙午平台（wp16）和终点（wp31）都做特殊处理
                if m_idx in [6, 16, 31]:
                    close_to_platform = (wp_idx == m_idx) & (dist_to_wp < 0.3)
                    root_vel = self._extract_root_state(data)[2]
                    speed_xy = np.linalg.norm(root_vel[:, :2], axis=1)
                    slow_enough = speed_xy < 0.15
                    trigger_celebration = close_to_platform & slow_enough & (~celebration_done)
                    if np.any(trigger_celebration):
                        state.info[done_key] = np.where(trigger_celebration, True, celebration_done)
                    if np.any((wp_idx == m_idx) & (~celebration_done)):
                        idxs = np.where((wp_idx == m_idx) & (~celebration_done))[0]
                        for i in idxs:
                            current_waypoints[i] = robot_xy[i]
                    block_advance |= (wp_idx == m_idx) & (~celebration_done)
                else:
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
        
        # 计算地形自适应目标速度
        wp_idx = state.info.get("hardcoded_waypoint_idx", np.zeros(data.shape[0], dtype=np.int32))
        target_speed = self._compute_target_speed(wp_idx)
        
        # 计算期望速度命令（参考目标速度）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 使用目标速度缩放期望速度
        desired_vel_norm = np.linalg.norm(desired_vel_xy, axis=1, keepdims=True)
        # 处理零向量情况，避免除以零
        mask = desired_vel_norm > 1e-6
        
        # 创建一个与desired_vel_xy相同形状的零向量
        zero_vel = np.zeros_like(desired_vel_xy)
        
        # 对于非零向量，计算单位向量并缩放
        # 使用更安全的方式计算，避免广播问题
        scaled_vel = np.zeros_like(desired_vel_xy)
        if np.any(mask):
            # 只对非零向量进行缩放
            idx = np.where(mask.flatten())[0]
            if len(idx) > 0:
                # 计算单位向量
                unit_vel = desired_vel_xy[idx] / desired_vel_norm[idx]
                # 缩放到目标速度
                scaled_vel[idx] = unit_vel * target_speed[idx, np.newaxis]
        
        # 使用缩放后的速度或零向量
        desired_vel_xy = np.where(mask, scaled_vel, zero_vel)
        
        # 确保结果有效
        desired_vel_xy = np.nan_to_num(desired_vel_xy, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
        # 到达里程碑后必须完成预设庆祝动作，完成后才允许继续前往下一路点
        milestone_ys = self._milestone_positions[:, 1].tolist()
        milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
        milestone_wp_idx = self._hardcoded_milestone_wp_idx.tolist()
        robot_x = root_pos[:, 0]
        robot_y = root_pos[:, 1]
        current_wp_idx = state.info.get("hardcoded_waypoint_idx", np.zeros(data.shape[0], dtype=np.int32))
        # v7.17: 出生保护期内不触发庆祝（先着陆站稳再说）
        in_spawn_grace = state.info.get("spawn_steps", np.full(data.shape[0], 999, dtype=np.int32)) < 100

        for milestone_y, milestone_name, m_wp_idx in zip(milestone_ys, milestone_names, milestone_wp_idx):
            done_key = f"{milestone_name}_celebration_done"
            hold_key = f"{milestone_name}_hold_steps"
            pose_key = f"{milestone_name}_pose_steps"
            start_key = f"{milestone_name}_celebration_started"

            if done_key not in state.info:
                state.info[done_key] = np.zeros(data.shape[0], dtype=bool)
            if hold_key not in state.info:
                state.info[hold_key] = np.zeros(data.shape[0], dtype=np.int32)
            if pose_key not in state.info:
                state.info[pose_key] = np.zeros(data.shape[0], dtype=np.int32)
            if start_key not in state.info:
                state.info[start_key] = np.zeros(data.shape[0], dtype=bool)

            on_milestone_wp = current_wp_idx == m_wp_idx
            at_milestone = self._in_milestone_zone(robot_x, robot_y, milestone_y)

            # 锁存触发：到达该里程碑窗口后进入庆祝，直到done前不退出
            # v7.17: 出生保护期(100步≈1秒)内不触发，让机器人先着陆站稳
            should_start = on_milestone_wp & at_milestone & (~state.info[done_key]) & (~in_spawn_grace)
            state.info[start_key] = np.where(
                state.info[done_key],
                False,
                state.info[start_key] | should_start,
            )
            celebration_active = state.info[start_key] & (~state.info[done_key])

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

            # 庆祝期间逐渐减速，避免突然刹车
            # 使用总庆祝步数（hold+pose）计算衰减系数，平滑过渡到静止
            total_celebration_steps = state.info[hold_key] + state.info[pose_key]
            max_decel_steps = 30  # 约0.3秒内减速到0（假设100Hz）
            decel_alpha = np.clip(1.0 - total_celebration_steps[celebration_active].astype(np.float32) / max_decel_steps, 0.0, 1.0)
            if np.any(celebration_active):
                desired_vel_xy[celebration_active, :] *= decel_alpha[:, np.newaxis]
                desired_yaw_rate[celebration_active] *= decel_alpha

            # 接近平台时减速（防止冲出平台）
            dist_to_milestone = np.sqrt((robot_x - self._milestone_positions[milestone_names.index(milestone_name), 0])**2 + (robot_y - milestone_y)**2)
            approaching_milestone = on_milestone_wp & (~celebration_active) & (~state.info[done_key]) & (dist_to_milestone < 3.0)
            if np.any(approaching_milestone):
                speed_limit = np.linalg.norm(desired_vel_xy[approaching_milestone, :], axis=1, keepdims=True)
                desired_vel_xy[approaching_milestone, :] = (
                    desired_vel_xy[approaching_milestone, :] / np.maximum(speed_limit, 1e-6) * 0.25
                )
                desired_yaw_rate[approaching_milestone] = desired_yaw_rate[approaching_milestone] * 0.3

            # 庆祝完成判定：姿势阶段保持够1.5秒
            pose_complete = in_pose & (state.info[pose_key] >= self._celebration_pose_steps)
            state.info[done_key] = state.info[done_key] | pose_complete
            state.info[start_key] = np.where(state.info[done_key], False, state.info[start_key])
        
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
        
        # 最远waypoint统计 + 更新全局前沿
        current_wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(data.shape[0], dtype=np.int32))
        max_wp_reached = np.max(current_wp_idx)
        self._global_max_wp = max(self._global_max_wp, int(max_wp_reached))  # 更新训练前沿
        
        # 楼梯通过率（到达 wp13 或更高，上楼梯+吊桥完成）
        current_wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(data.shape[0], dtype=np.int32))
        stair_pass_ratio = np.mean(current_wp_idx >= 13)
        
        # 下楼梯完成率（到达 wp16 或更高，稳定到达丙午平台）
        descent_pass_ratio = np.mean(current_wp_idx >= 16)
        
        # 卡住比例
        stuck_ratio = np.mean(info.get("stuck_counter", np.zeros(data.shape[0], dtype=np.int32)) > 100)
        
        # 添加到窗口
        if len(self.metric_window['robot_y']) < self.metric_window_size:
            self.metric_window['robot_y'].append(avg_y)
            self.metric_window['rewards'].append(avg_reward)
            self.metric_window['stair_pass'].append(stair_pass_ratio)
            self.metric_window['descent_pass'].append(descent_pass_ratio)
            self.metric_window['stuck_ratio'].append(stuck_ratio)
            self.metric_window['max_waypoint'].append(max_wp_reached)
        else:
            # 滑动窗口
            self.metric_window['robot_y'] = self.metric_window['robot_y'][1:] + [avg_y]
            self.metric_window['rewards'] = self.metric_window['rewards'][1:] + [avg_reward]
            self.metric_window['stair_pass'] = self.metric_window['stair_pass'][1:] + [stair_pass_ratio]
            self.metric_window['descent_pass'] = self.metric_window['descent_pass'][1:] + [descent_pass_ratio]
            self.metric_window['stuck_ratio'] = self.metric_window['stuck_ratio'][1:] + [stuck_ratio]
            self.metric_window['max_waypoint'] = self.metric_window['max_waypoint'][1:] + [max_wp_reached]
        
        # 每1000步打印一次诊断信息
        if self.navigation_stats_step % 500 == 0:
            self._print_training_metrics()
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        state.info = info
        
        return state
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        终止条件:
        1. IMU侧翻检测: |Roll|>60° 或 |Pitch|>60° (替代旧版gravity_z检测)
        2. 超出边界 (Y>各阶段设定 或 Y<-3.5掉下悬崖)
        3. 物理发散 (NaN检测)
        4. 卡住不动 (连续450步速度<0.05且未到达终点)
        """
        data = state.data
        info = state.info
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # === 1. IMU Roll/Pitch侧翻检测（替代旧版gravity_z方案） ===
        # 直接从四元数解算Roll/Pitch角，更精确地检测侧翻和俯仰
        roll, pitch = self._compute_roll_pitch_from_quat(root_quat)
        roll_pitch_threshold = np.deg2rad(self.tilt_threshold_deg)  # 使用配置的侧翻阈值
        tilt_terminated = (np.abs(roll) > roll_pitch_threshold) | (np.abs(pitch) > roll_pitch_threshold)
        
        # 保留gravity_z计算用于奖励系统的软倾斜惩罚（不用于终止）
        # gravity_z在reward中仍然使用
        
        y_out_forward = root_pos[:, 1] > self.boundary_y_max
        y_fall_backward = root_pos[:, 1] < -3.4  # 后退掉落悬崖（远离出生点）
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

        # v7.17: 庆祝动作期间不算"卡住"（庆祝要求静止3秒是正常的）
        is_celebrating = np.zeros(root_pos.shape[0], dtype=bool)
        for m_name in ["milestone_2026", "milestone_bingwu", "milestone_final"]:
            start_key = f"{m_name}_celebration_started"
            if start_key in info:
                is_celebrating |= info[start_key]
        info["stuck_counter"] = np.where(is_celebrating, 0, info["stuck_counter"])
        
        # 获取当前waypoint索引，为wp27圆盘石头区域增加更多尝试机会
        current_wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(root_pos.shape[0], dtype=np.int32))
        in_disc_area = (current_wp_idx >= 21) & (current_wp_idx <= 28)  # wp21-28区域（圆盘石头区）
        
        # 设置不同的卡住阈值：普通区域480步，圆盘石头区650步
        stuck_threshold = np.where(in_disc_area, 650, 480)
        
        # 卡住超过阈值且未到达终点就终止
        reached_goal = root_pos[:, 1] >= self.goal_y
        stuck_terminated = info["stuck_counter"] > stuck_threshold
        # 到达终点后不因卡住终止，让机器人保持庆祝动作
        stuck_terminated = stuck_terminated & (~reached_goal)
        
        # 合并所有终止条件
        # 到达终点后只因为侧翻、超出边界或物理发散而终止
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
        print(f"📊【训练进度】步数: {self.navigation_stats_step:,}  课程前沿: wp{self._global_max_wp}", file=sys.stderr)
        print("="*90, file=sys.stderr)
        
        total_wp = len(self._hardcoded_waypoints)  # v7.9: 动态获取waypoint总数
        print(f"【航点】最远达到: wp{max_wp}/{total_wp} ({max_wp/total_wp*100:.1f}%)  课程出生偏重: wp{max(0,self._global_max_wp-3)}~wp{min(total_wp-1,self._global_max_wp+3)}", file=sys.stderr)
        if max_wp >= total_wp:
            print("       ✅ 完成全程", file=sys.stderr)
        elif max_wp >= 17:  # v7.9: 调整里程碑判断(16=丙午平台)
            print("       ✅ 已过楼梯", file=sys.stderr)
        elif max_wp >= 8:
            print("       ⚠️  进入楼梯", file=sys.stderr)
        elif max_wp >= 7:
            print("       ⚠️  到达2026平台", file=sys.stderr)
        else:
            print("       ❌ 尚在坑洼区", file=sys.stderr)
        
        print(f"【进度】avg_y={avg_y:.2f}m  增速={y_improve:.4f}m/step", file=sys.stderr)
        if y_improve > 0.002:
            print("       ✅ 快速进步（机器人越走越远）", file=sys.stderr)
        elif y_improve > 0.0005:
            print("       ⚠️  缓慢进步（在学习但提升不大）", file=sys.stderr)
        else:
            print("       ❌ 停滞（机器人没有走得更远）", file=sys.stderr)
        
        print(f"【奖励】avg={avg_reward:.2f}分  改进={reward_improve:.4f}/step", file=sys.stderr)
        print(f"        ↳ 奖励=每步综合得分，含速度跟踪+前进+waypoint等", file=sys.stderr)
        if avg_reward > 50:
            print("       ✅ 学得好（每步综合得分高，策略有效）", file=sys.stderr)
        elif avg_reward > 10:
            print("       ⚠️  在学（有正向奖励，策略在摸索中）", file=sys.stderr)
        elif avg_reward > 0:
            print("       ⚠️  微弱学习（正奖励很少，可能在摔倒/卡住）", file=sys.stderr)
        else:
            print("       ❌ 无学习信号（总奖励为负=惩罚>奖励，策略还在乱走）", file=sys.stderr)
        
        print(f"【楼梯】上楼+吊桥(wp≥13)={avg_stair*100:.1f}%", file=sys.stderr)
        if avg_stair > 0.8:
            print("       ✅ 完全掌握", file=sys.stderr)
        elif avg_stair > 0.5:
            print("       ⚠️  基本通过", file=sys.stderr)
        elif avg_stair > 0.2:
            print("       ⚠️  部分通过", file=sys.stderr)
        else:
            print("       ❌ 卡住", file=sys.stderr)
        
        avg_descent = np.mean(self.metric_window['descent_pass'])
        print(f"【下楼】下楼梯→丙午(wp≥16)={avg_descent*100:.1f}%", file=sys.stderr)
        if avg_descent > 0.6:
            print("       ✅ 稳定下楼", file=sys.stderr)
        elif avg_descent > 0.3:
            print("       ⚠️  部分成功", file=sys.stderr)
        elif avg_descent > 0.1:
            print("       ⚠️  偶尔成功", file=sys.stderr)
        else:
            print("       ❌ 下楼困难", file=sys.stderr)
        
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
        
        # ========== 专家级奖励系统 v7.5 ==========
        # v7.5修复：机器人默认朝+X，但目标在+Y → v7.4硬编码vy导致方向错误
        # 关键改动：前进奖励改为dot(vel, goal_dir)投影，加回弱朝向对齐
        
        # --- 计算目标方向（用于前进投影+距离势函数） ---
        pose_commands = info["pose_commands"]
        goal_xy = pose_commands[:, :2]
        goal_dir = goal_xy - robot_xy
        goal_dist = np.linalg.norm(goal_dir, axis=1, keepdims=True)
        goal_dir_unit = goal_dir / np.maximum(goal_dist, 1e-6)
        
        # === 地形自适应参数（全部用wp_idx判断） ===
        robot_x = root_pos[:, 0]
        wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(num_envs, dtype=np.int32))
        self._cached_wp_idx = wp_idx
        # 波浪/坑洼地形：wp0~wp2
        # 2026平台前减速区：wp5~wp6
        # 上楼梯区：wp7~wp11
        # 下楼梯/吊桥区：wp13~wp15
        in_wave_terrain = (wp_idx >= 0) & (wp_idx <= 2)
        in_approach_2026 = (wp_idx >= 5) & (wp_idx <= 6)
        in_stairs_up = (wp_idx >= 8) & (wp_idx <= 10)
        in_bridge = (wp_idx >= 10) & (wp_idx <= 13)
        in_stairs_down = (wp_idx >= 13) & (wp_idx <= 15)
        in_reverse_stairs_up = (wp_idx >= 17) & (wp_idx <= 19)
        in_riverbed = (wp_idx >= 20) & (wp_idx <= 27)


        # 地形自适应目标速度
        target_speed = self._compute_target_speed(wp_idx)
        
        robot_z = root_pos[:, 2]  # 质心高度
        vel_z = root_vel[:, 2]    # 垂直速度
                # === 下楼梯/刹车时前腿主动刹车奖励 ===
        # 检测后腿离地且前腿着地，鼓励前腿输出大力矩保持平衡
        # 计算速度（提前计算，用于刹车判断）
        vel_xy = root_vel[:, :2]
        speed = np.linalg.norm(vel_xy, axis=1)  # 速度大小，任意方向

        # 关节索引: [0-2]=FR, [3-5]=FL, [6-8]=RR, [9-11]=RL
        # 简化判据：后腿力矩总和<5，前腿力矩总和>10，且在下楼梯或速度<0.2（刹车）
        actuator_forces = np.abs(data.actuator_ctrls)  # [num_envs, 12]
        front_leg_forces = np.sum(actuator_forces[:, 0:6], axis=1)  # FR+FL
        rear_leg_forces = np.sum(actuator_forces[:, 6:12], axis=1) # RR+RL
        # 增加靠近平台的区域：wp5-6和wp15-17
        in_approach_platform = (wp_idx >= 5) & (wp_idx <= 6) | (wp_idx >= 15) & (wp_idx <= 17)
        braking = (in_stairs_down | in_approach_platform | in_riverbed | (speed < 0.6))
        rear_off_ground = (rear_leg_forces < 5.0)
        front_brake = (front_leg_forces > 8.0)
        front_brake_reward = np.where(braking & rear_off_ground & front_brake, 0.20, 0.0)
        reward += front_brake_reward
        # 记录刹车状态，持续刹车给予额外奖励
        if "braking_steps" not in info:
            info["braking_steps"] = np.zeros(num_envs, dtype=np.int32)

        info["braking_steps"] = np.where(braking & rear_off_ground & front_brake, 
                                 info["braking_steps"] + 1, 
                                 0)

        # 持续刹车奖励
        sustained_brake_reward = np.where(info["braking_steps"] > 3, 0.5, 0.0)
        reward += sustained_brake_reward
        
        # 1. 质心高度稳定性奖励（波浪地形关键）
        # 记录上一步质心高度，惩罚高度剧烈变化
        if "prev_robot_z" not in info:
            info["prev_robot_z"] = robot_z.copy()
        z_change = np.abs(robot_z - info["prev_robot_z"])
        
        # 波浪地形时特别关注质心稳定（Y=0~4）
        # 同时关注 wp 6-7 之间和 wp 30-31 之间的区域
        in_critical_areas = in_wave_terrain | ((wp_idx >= 6) & (wp_idx <= 7)) | ((wp_idx >= 30) & (wp_idx <= 31))
        z_stability_reward = np.where(
            in_critical_areas,
            0.03 * np.exp(-5.0 * z_change),  # 关键区域：高度变化越小奖励越高
            0.01 * np.exp(-3.0 * z_change)   # 其他地形：较低权重
        )
        reward += z_stability_reward
        info["prev_robot_z"] = robot_z.copy()
        
        # 2. 升降动力学补偿（楼梯地形关键）
        # 上楼梯：允许适度向上速度，惩罚向下
        # 下楼梯：允许适度向下速度，惩罚过快下降
        stairs_dynamics_reward = np.where(
            in_stairs_up & (~in_reverse_stairs_up),
            # 上楼梯：奖励向上速度(0~0.6m/s)，惩罚向下
            0.05 * np.clip(vel_z, 0, 0.6) - 0.1 * np.clip(-vel_z, 0, 0.6),
            np.where(
                in_stairs_down,
                # 下楼梯：允许更慢下降(-0.2~0m/s)，惩罚过快下降
                -0.16 * np.clip(-vel_z - 0.15, 0, 0.6),
                0.0
            )
        )
        reward += stairs_dynamics_reward
        
        # 3. 俯仰角稳定性增强（波浪+楼梯共用）
        # 使用IMU数据估计pitch角速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        pitch_rate = np.abs(gyro[:, 1])  # Pitch角速度
        
        # v7.19: 波浪地形和楼梯段惩罚俯仰剧烈变化（阈值改为35°≈0.61rad）
        pitch_stability_threshold = np.deg2rad(35.0)  # 约0.61rad/s
        pitch_stability_penalty = np.where(
            in_critical_areas | in_stairs_up | in_stairs_down | in_reverse_stairs_up,
            -0.1 * np.clip(pitch_rate - pitch_stability_threshold, 0, 2.0),
            0.0
        )
        reward += pitch_stability_penalty
        
        # === v7.13 河床凹地形跨越策略（赛事方建议） ===
        # 问题1: 凹陷尺寸比例差异 → 根据质心高度变化判断凹陷深度
        # 问题2: 边缘悬崖效应 → 监控质心高度急剧下降
        # 问题3: 过渡段步态切换 → 平滑关节速度变化
        
        if in_riverbed.any():
            # 1. 凹陷边缘检测奖励（避免足端悬空）
            # 质心高度急剧下降(>0.1m/step)表示可能踩空边缘
            edge_fall_risk = z_change > 0.1
            edge_penalty = np.where(
                in_riverbed & edge_fall_risk,
                -0.1,  # 边缘悬空风险惩罚
                0.0
            )
            reward += edge_penalty
            
            # 2. 步态切换平滑奖励（减少绊倒/卡顿）
            # 监控关节速度变化率（角加速度）
            joint_vel = self.get_dof_vel(data)  # [num_envs, 12]
            if "prev_joint_vel" not in info:
                info["prev_joint_vel"] = joint_vel.copy()
            joint_accel = joint_vel - info["prev_joint_vel"]
            joint_jerk = np.sum(joint_accel**2, axis=1)  # 关节加速度平方和
            
            # 河床区域奖励平滑步态切换
            gait_smooth_reward = np.where(
                in_riverbed,
                0.2 * np.exp(-0.5 * joint_jerk),  # 关节变化越平滑奖励越高
                0.0
            )
            reward += gait_smooth_reward
            info["prev_joint_vel"] = joint_vel.copy()
            
            # 3. 跨越成功奖励（鼓励快速通过凹陷区）
            # 在河床区域保持前进速度
            riverbed_forward_bonus = np.where(
                in_riverbed & (speed > 0.4),  # 速度>0.4m/s
                0.3,  # 持续前进奖励
                0.0
            )
            reward += riverbed_forward_bonus
        
        # 1. 速度大小跟踪（地形自适应目标速度）
        # vel_xy和speed已在上面计算
        speed_error = np.square(speed - target_speed)  # 自适应目标速度
        tracking_speed = np.exp(-speed_error / 0.25)
        reward += 1.0 * tracking_speed  # 跑得快就有奖励，不管往哪跑
        
        # 2. 前进投影（v7.5修复：从section001移植dot(vel, target_dir)，不再硬编码vy）
        # 机器人朝+X但目标在+Y，用投影才能正确奖励朝目标方向的速度
        vel_toward_goal = np.sum(vel_xy * goal_dir_unit.squeeze(), axis=1)
        forward_progress = np.clip(vel_toward_goal, -0.5, 1.5)
        
        # 地形自适应前进奖励倍数
        # 上楼梯：高奖励鼓励前进(1.8x)
        # 下楼梯/吊桥：低倍数避免冲太快(0.8x)
        # 接近2026平台：减速接近(0.9x)
        # 其他：正常(1.2x)
        terrain_forward_bonus = np.where(
            in_stairs_up, 1.9,
            np.where(in_stairs_down, 0.60,
            np.where(in_approach_2026, 0.9, 1.2))
        )
        reward += terrain_forward_bonus * forward_progress
        
        # 3. 弱朝向对齐（帮助初始转向：机器人朝+Y时目标也在+Y，很容易对齐）
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(goal_dir[:, 1], goal_dir[:, 0]).flatten()
        heading_error = desired_heading - robot_heading
        heading_error = np.where(heading_error > np.pi, heading_error - 2*np.pi, heading_error)
        heading_error = np.where(heading_error < -np.pi, heading_error + 2*np.pi, heading_error)
        heading_align = np.exp(-np.square(heading_error / 0.6))
        reward += 0.25 * heading_align  # 降低到0.2，不压制横向收集地标
        
        # v7.6: 移除角速度限制，让机器人可以自由转动
        # 卡住时多转转可能找到新路径，不应该限制yaw_rate
        
        # 4. Sigmoid柔性距离追踪（赛事方建议：FLEXIBLE TRACKING）
        # R = 1/(1+e^(k*d))，k=0.5，d=当前距离/初始距离
        # 优势：平滑衰减、允许探索、降低僵硬
        dist_to_goal = goal_dist.flatten()
        
        # 记录初始距离（用于计算距离比例）
        if "init_dist_to_goal" not in info:
            info["init_dist_to_goal"] = np.maximum(dist_to_goal.copy(), 1.0)  # 至少1m防止除零
        
        # 距离比例 d = current_dist / init_dist
        dist_ratio = dist_to_goal / info["init_dist_to_goal"]
        dist_ratio = np.clip(dist_ratio, 0.0, 2.0)  # 限制最大比例防止跃开
        
        # Sigmoid柔性追踪奖励: R = 1/(1+e^(k*d))
        k_sigmoid = 0.5  # 距离系数
        sigmoid_reward = 1.0 / (1.0 + np.exp(k_sigmoid * dist_ratio * 5))  # 调整范围至[0,1]
        reward += 0.3 * sigmoid_reward  # 较小权重，不过度约束
        
        # 保留一个小的距离改进奖励（但降低权重）
        if "prev_dist_to_goal" not in info:
            info["prev_dist_to_goal"] = dist_to_goal.copy()
        dist_improvement = info["prev_dist_to_goal"] - dist_to_goal
        goal_shaping = np.clip(dist_improvement * 1.0, -0.05, 0.1)  # v7.10: 降低权重
        reward += goal_shaping
        info["prev_dist_to_goal"] = dist_to_goal.copy()
        
        # 6. 存活奖励（鼓励不摔倒）
        reward += 0.05
        
        # 7. 新区域探索奖励（到达距目标更近的位置时给一次性奖励）
        if "min_dist_to_goal" not in info:
            info["min_dist_to_goal"] = dist_to_goal.copy()
        new_territory = dist_to_goal < info["min_dist_to_goal"] - 0.2  # 比之前最近再近0.5m
        reward += new_territory.astype(np.float32) * 1.0  # 每次突破+1.0
        info["min_dist_to_goal"] = np.minimum(info["min_dist_to_goal"], dist_to_goal)
        
        # 7b. Waypoint递进奖励（到达新waypoint给递增奖励：基准15.0 + 序号×2.0）
        # 越远的waypoint奖励越大，鼓励向前推进
        wp_idx = info.get("hardcoded_waypoint_idx", np.zeros(num_envs, dtype=np.int32))
        if "prev_waypoint_idx" not in info:
            info["prev_waypoint_idx"] = np.zeros(num_envs, dtype=np.int32)
        new_wp_reached = wp_idx > info["prev_waypoint_idx"]
        if np.any(new_wp_reached):
            # 到达奖励 = 基准值(15.0) + 序号递增(2.0×idx)
            wp_reward = 15.0 + wp_idx[new_wp_reached].astype(np.float32) * 2.0
            reward[new_wp_reached] += wp_reward
        info["prev_waypoint_idx"] = wp_idx.copy()
        
        # 8. 软倾斜惩罚（渐进式：坑洼地形倾斜是正常的，不应终止）
        projected_gravity = self._compute_projected_gravity(root_quat)
        gravity_z = projected_gravity[:, 2]  # 正常站立时≈-1，完全倒下时≈0
        # tilt_angle ≈ arccos(-gravity_z)：0°=站直，90°=侧翻
        tilt_cos = np.clip(-gravity_z, 0.0, 1.0)  # 0=倒下, 1=站直
        # 45°以内无惩罚，45-60°渐进惩罚，60°以上终止(Roll/Pitch检测)
        
        # v7.14: 软倾斜惩罚阈值降低45°→35°（cos35°≈0.82），增强权重
        soft_tilt_penalty = np.where(
            tilt_cos > 0.82,  # cos(35°)≈0.82, tilt<35°: 无惩罚
            0.0,
            -0.05 * (0.82 - tilt_cos) / 0.82  # 加大惩罚：0.02→0.05
        )
        # 楼梯段加强稳定性惩罚（倾斜太大就重罚）
        in_stairs = in_stairs_up | in_stairs_down
        stair_stability_penalty = np.where(
            in_stairs & (tilt_cos < 0.6),  # 楼梯上倾斜>50°就重罚
            -0.2 * (0.6 - tilt_cos),  # 加大惩罚：0.1→0.2
            0.0
        )
        # 下楼梯额外稳定性惩罚（训练显示下楼梯太快容易摔）
        descent_speed_penalty = np.where(
            in_stairs_down & (speed > 0.3),  # v7.15: 下楼梯阈值0.6->0.38
            -0.2 * (speed - 0.3),  # 加大惩罚：0.12→0.2
            0.0
        )
        # 下楼梯超速时更强的倾斜惩罚
        descent_tilt_penalty = np.where(
            in_stairs_down & (tilt_cos < 0.7),  # 下楼梯倾斜>45°就重罚
            -0.3 * (0.7 - tilt_cos),  # 加大惩罚：0.15→0.3
            0.0
        )
        reward += soft_tilt_penalty + stair_stability_penalty + descent_speed_penalty + descent_tilt_penalty
        
        # === 关节姿态自然性奖励（v7.10） ===
        # 问题：后腿经常伸得很直（calf接近0而不是-1.8），步态不自然且容易卡腿
        # 解决：奖励calf关节保持在自然弯曲范围内
        # 关节索引: 0=FR_hip, 1=FR_thigh, 2=FR_calf, 3=FL_hip, 4=FL_thigh, 5=FL_calf,
        #            6=RR_hip, 7=RR_thigh, 8=RR_calf, 9=RL_hip, 10=RL_thigh, 11=RL_calf
        joint_pos = self.get_dof_pos(data)  # [num_envs, 12]
        
        # calf关节索引: 2, 5, 8, 11
        calf_indices = [2, 5, 8, 11]
        calf_positions = joint_pos[:, calf_indices]  # [num_envs, 4]
        
        # 自然calf位置约-1.8，接近0表示过度伸展
        # 奖励calf保持在[-2.4, -0.6]范围内（允许一定活动范围）
        calf_natural_min = -2.4
        calf_natural_max = -0.6
        
        # 计算calf过度伸展的惩罚（calf > -0.6表示伸得太直）
        calf_overextend = np.maximum(calf_positions - calf_natural_max, 0)  # 超过-0.6的部分
        calf_overextend_penalty = -0.02 * np.sum(calf_overextend, axis=1)  # 每个关节贡献惩罚
        
        # 计算calf过度弯曲的惩罚（calf < -2.4表示弯得太深）
        calf_overbend = np.maximum(calf_natural_min - calf_positions, 0)
        calf_overbend_penalty = -0.01 * np.sum(calf_overbend, axis=1)
        
        # 后腿特别惩罚（后腿更容易伸直）
        rear_calf_positions = joint_pos[:, [8, 11]]  # RR_calf, RL_calf
        rear_calf_overextend = np.maximum(rear_calf_positions - (-1.2), 0)  # 后腿更严格: -1.2
        rear_calf_penalty = -0.05 * np.sum(rear_calf_overextend, axis=1)
        
        reward += calf_overextend_penalty + calf_overbend_penalty + rear_calf_penalty

        # v7.15: 上楼梯后腿接地/发力奖励（缓解“前腿拉、后腿悬空”）
        actuator_forces_abs = np.abs(data.actuator_ctrls)
        rear_leg_effort = np.sum(actuator_forces_abs[:, 6:12], axis=1)
        rear_calf_bent = (joint_pos[:, 8] < -1.0) & (joint_pos[:, 11] < -1.0)
        rear_support_reward = np.where(
            in_stairs_up & rear_calf_bent & (rear_leg_effort > 12.0),
            0.12,
            0.0,
        )
        reward += rear_support_reward
        
        # === 姿态稳定性控制（v7.11 赛事方建议） ===
        # 1. 姿态平滑性奖励：惩罚角速度过大（Roll/Pitch方向）
        # gyro[:, 0]=Roll角速度, gyro[:, 1]=Pitch角速度, gyro[:, 2]=Yaw角速度
        roll_rate = np.abs(gyro[:, 0])   # Roll角速度
        pitch_rate = np.abs(gyro[:, 1])  # Pitch角速度
        
        # v7.14: 增强姿态平滑权重（稳定性信号需要和前进信号平衡）
        # 角速度越小奖励越高，exp衰减让小角速度获得高奖励
        attitude_smoothness = np.exp(-0.8 * (roll_rate**2 + pitch_rate**2))  # 更陡峭的衰减
        reward += 0.15 * attitude_smoothness  # v7.14: 0.05→0.15 增强稳定性权重
        
        # 2. 角加速度感知（趋势预测）：惩罚角速度突变
        if "prev_gyro" not in info:
            info["prev_gyro"] = gyro.copy()
        angular_accel = gyro - info["prev_gyro"]  # 角加速度 ≈ Δω/Δt
        angular_jerk = np.sum(angular_accel[:, :2]**2, axis=1)  # Roll+Pitch的角加速度平方和
        # v7.14: 降低阈值0.1→0.05，增强惩罚0.02→0.08
        jerk_penalty = -0.08 * np.clip(angular_jerk - 0.05, 0, 1.0)
        reward += jerk_penalty
        info["prev_gyro"] = gyro.copy()
        
        # 3. 足端接触对称性奖励（鼓励四足平衡着地）
        # 从actuator_ctrls估计各腿的"着地力"（简化：大力矩≈着地）
        # 关节索引: [0-2]=FR, [3-5]=FL, [6-8]=RR, [9-11]=RL
        actuator_forces = np.abs(data.actuator_ctrls)  # [num_envs, 12]
        # 每条腿的总力矩（3个关节求和）
        leg_forces = np.stack([
            np.sum(actuator_forces[:, 0:3], axis=1),   # FR
            np.sum(actuator_forces[:, 3:6], axis=1),   # FL  
            np.sum(actuator_forces[:, 6:9], axis=1),   # RR
            np.sum(actuator_forces[:, 9:12], axis=1),  # RL
        ], axis=1)  # [num_envs, 4]
        
        # 计算左右对称性（FR+RR vs FL+RL）
        right_total = leg_forces[:, 0] + leg_forces[:, 2]  # FR + RR
        left_total = leg_forces[:, 1] + leg_forces[:, 3]   # FL + RL
        lr_asymmetry = np.abs(right_total - left_total) / (np.abs(right_total + left_total) + 1e-6)
        
        # 计算前后对称性（FR+FL vs RR+RL）
        front_total = leg_forces[:, 0] + leg_forces[:, 1]  # FR + FL
        rear_total = leg_forces[:, 2] + leg_forces[:, 3]   # RR + RL
        fr_asymmetry = np.abs(front_total - rear_total) / (np.abs(front_total + rear_total) + 1e-6)
        
        # v7.14: 对称性奖励（越对称奖励越高）增强权重
        symmetry_reward = 0.08 * (1.0 - 0.5 * lr_asymmetry - 0.5 * fr_asymmetry)
        reward += symmetry_reward

        # 3b. 楼梯卡住惩罚（v7.17 重写，替代v7.16通用碰撞检测）
        # 目的：惩罚在楼梯区进退两难（高力矩+低速度=推着台阶走不动）
        # 关键：仅在楼梯/吊桥区域 + 速度极低时触发，正常行走完全不受影响
        # 原v7.16的问题：actuator_ctrls是关节力矩(正常走路thigh>10Nm)，不是接触力，
        #   force_total>5阈值正常走路就超了→每步都判碰撞→累积器永不归零→avg_reward=-4.45
        # 卡住检测区域扩展：上楼梯、下楼梯、吊桥、波浪/坑洼地形（wp0~2）、wp6~7之间
        in_wave_terrain_stuck = in_wave_terrain
        in_67_stuck = (wp_idx >= 6) & (wp_idx <= 7)
        in_stuck_zone = in_stairs_up | in_stairs_down | in_bridge | in_wave_terrain_stuck | in_67_stuck
        stair_stuck_speed_thresh = 0.08  # 速度<0.08m/s才算卡住
        stair_stuck = in_stuck_zone & (speed < stair_stuck_speed_thresh)

        # 力矩饱和度：当关节力矩接近限制但速度仍然很低，说明被阶梯卡住
        # hip/thigh限制17Nm, calf限制34Nm → 归一化到[0,1]
        torque_limits_vec = np.array([17, 17, 34] * 4, dtype=np.float32)
        torque_saturation = np.mean(np.abs(data.actuator_ctrls) / torque_limits_vec, axis=1)  # [0~1]
        # 力矩饱和度 > 0.5 表示大力输出却走不动 → 吊桥更容易卡，阈值略降
        stair_stuck_confirmed = stair_stuck & (torque_saturation > 0.5)

        # 楼梯卡住计数器（渐进惩罚：越卡越重）
        if "stair_stuck_counter" not in info:
            info["stair_stuck_counter"] = np.zeros(num_envs, dtype=np.int32)
        info["stair_stuck_counter"] = np.where(
            stair_stuck_confirmed,
            info["stair_stuck_counter"] + 1,
            np.maximum(info["stair_stuck_counter"] - 5, 0),  # 脱困后快速恢复
        )
        # 渐进惩罚：前30步(0.3秒)无惩罚，之后每步-0.08，上限每步-0.5（吊桥更快惩罚）
        stair_stuck_steps = info["stair_stuck_counter"].astype(np.float32)
        stair_stuck_penalty = -np.clip((stair_stuck_steps - 30) * 0.08, 0.0, 0.5)
        reward += np.where(stair_stuck_confirmed, stair_stuck_penalty, 0.0)
        
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
        
        # ========== 4. 多里程碑平台奖励（3个平台） ==========
        # 里程碑: Y=8(2026平台), Y=24(丙午大吉), Y=32(终点)
        milestone_ys = self._milestone_positions[:, 1].tolist()
        milestone_names = ["milestone_2026", "milestone_bingwu", "milestone_final"]
        
        for milestone_y, milestone_name in zip(milestone_ys, milestone_names):
            milestone_key = f"{milestone_name}_reached"
            if milestone_key not in info:
                info[milestone_key] = np.zeros(num_envs, dtype=bool)
            
            # 到达里程碑判断：Y在±0.4内 且 X在±3内
            reached_milestone = self._in_milestone_zone(robot_x, robot_y, milestone_y)
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
            at_milestone = self._in_milestone_zone(robot_x, robot_y, milestone_y)
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
            start_key = f"{milestone_name}_celebration_started"
            
            if celebration_done_key not in info:
                info[celebration_done_key] = np.zeros(num_envs, dtype=bool)
            if pose_key not in info:
                info[pose_key] = np.zeros(num_envs, dtype=np.int32)
            
            # 在平台窗口内或已锁存庆祝阶段，都算庆祝进行中
            at_milestone = self._in_milestone_zone(robot_x, robot_y, milestone_y)
            celebration_active = info.get(start_key, np.zeros(num_envs, dtype=bool))
            active_or_on_platform = at_milestone | celebration_active
            
            # 姿势阶段奖励：关节角度接近目标预设越近奖励越大
            in_pose = active_or_on_platform & (~info[celebration_done_key]) & (info.get(pose_key, np.zeros(num_envs, dtype=np.int32)) > 0)
            if np.any(in_pose):
                # 计算当前关节角度与目标姿势的距离
                joint_error = np.linalg.norm(joint_pos[in_pose] - self._celebration_pose, axis=1)
                # 奖励接近目标姿势（误差越小奖励越大，配合动作覆盖确保姿势到位）
                pose_reward = np.exp(-joint_error * 2.0) * 0.5  # 每步最大≈+0.5（覆盖后基本满分）
                reward[in_pose] += pose_reward
            
            # 庆祝完成奖励：姿势保持够时间就给大奖励
            if self.enable_celebration_reward:
                celebration_complete = (
                    active_or_on_platform
                    & (~info[celebration_done_key])
                    & (info.get(pose_key, np.zeros(num_envs, dtype=np.int32)) >= self._celebration_pose_steps)
                )
                reward += celebration_complete.astype(np.float32) * self.celebration_reward
                info[celebration_done_key] = info[celebration_done_key] | celebration_complete
        
        # ========== 6. 稳定性惩罚 ==========
        # v7.1: 移除角速度惩罚——大步快跑时陀螺仪值大是正常的
        # 已有倾斜检测(>70°终止)防止摔倒，不需要额外惩罚角速度
        
        # 终止惩罚（使用Roll/Pitch检测，与_compute_terminated保持一致）
        roll, pitch = self._compute_roll_pitch_from_quat(root_quat)
        roll_pitch_threshold = np.deg2rad(self.tilt_threshold_deg)  # 使用配置的侧翻阈值
        tilt_terminated = (np.abs(roll) > roll_pitch_threshold) | (np.abs(pitch) > roll_pitch_threshold)
        reward += tilt_terminated.astype(np.float32) * (-60.0)  # 加大惩罚：30→60
        
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
        
        # === 课程学习v3：前沿偏重出生（Frontier-Biased Curriculum） ===
        # 跟踪global_max_wp（训练中最远到达的wp），自动集中资源在前沿
        # 30%环境：从起点出发（防遗忘）
        # 70%课程环境中：
        #   60% 在前沿附近出生（前沿wp ± 3个wp范围内）→ 专攻难点
        #   40% 在已学区域随机出生（0 ~ 前沿wp）→ 巩固复习
        num_curriculum = int(num_envs * self._curriculum_spawn_fraction)
        self._curriculum_start_wp_idx = np.zeros(num_envs, dtype=np.int32)  # 记录每个env的起始wp
        if num_curriculum > 0:
            cur_idx = np.arange(num_curriculum)
            max_wp = len(self._hardcoded_waypoints) - 1
            frontier = min(self._global_max_wp, max_wp)
            
            # 分配：60%前沿 + 40%复习
            num_frontier = int(num_curriculum * 0.6)
            num_review = num_curriculum - num_frontier
            
            # 前沿出生：在 [frontier-3, frontier+3] 范围内随机（聚焦难点）
            frontier_low = max(0, frontier - 3)
            frontier_high = min(max_wp, frontier + 3)
            frontier_wp = np.random.randint(frontier_low, frontier_high + 1, size=num_frontier)
            
            # 复习出生：在 [0, frontier] 范围内随机（巩固已学）
            review_wp = np.random.randint(0, max(1, frontier + 1), size=num_review)
            
            # 合并
            spawn_wp_indices = np.concatenate([frontier_wp, review_wp])
            spawn_wp_positions = self._hardcoded_waypoints[spawn_wp_indices]  # [num_curriculum, 2]
            
            # 在waypoint附近加小随机偏移（±0.3m）
            spawn_jitter = np.random.uniform(-0.3, 0.3, size=(num_curriculum, 2))
            robot_init_xy[cur_idx, 0] = spawn_wp_positions[:, 0] + spawn_jitter[:, 0]
            robot_init_xy[cur_idx, 1] = spawn_wp_positions[:, 1] + spawn_jitter[:, 1]
            terrain_heights[cur_idx] = self._curriculum_spawn_z  # 高处出生（2.0m），安全着陆
            
            # 记录每个课程环境的起始waypoint索引
            self._curriculum_start_wp_idx[cur_idx] = spawn_wp_indices
        
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
        
        # ========== 初始朝向随机化（官方要求）==========
        # v7.18: 360°完全随机朝向（官方规则）
        random_yaw = np.random.uniform(-np.pi, np.pi, size=num_envs)
        
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
        assert obs.shape == (num_envs, 67)  # 54 + 13 = 67维
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "spawn_steps": np.zeros(num_envs, dtype=np.int32),  # v7.17: 出生保护期计数
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
        }
        
        # 硬编码导航初始化：课程学习环境从对应waypoint开始，非课程从wp0开始
        if self.use_hardcoded_navigation:
            info["hardcoded_waypoint_idx"] = self._curriculum_start_wp_idx.copy()
        
        return obs, info