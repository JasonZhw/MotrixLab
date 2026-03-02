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

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class ControlConfig:
    # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
    # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
    action_scale = 0.25  # 平地navigation使用0.25
    # torque_limit[N*m] 使用XML forcerange参数

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.5]  
    
    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

    # the default angles for all joints. key = joint name, value = target angle [rad]
    # 使用locomotion的关节角度配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }
#四足12个维度 髋关节 大腿 小腿 每条腿3个关节，共4条腿12个关节角度
@dataclass
class Commands:
    # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 相对机器人初始位置的偏移（米）
    # yaw: 目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05

@dataclass
class Asset:
    body_name = "base"
    foot_names = ["FR", "FL", "RR", "RL"]
    terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
    ground_subtree = "C_"  # 地形根节点，用于subtree接触检测
   
@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"
    feet = ["FR", "FL", "RR", "RL"]  # 足部接触力传感器名称

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
            "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
            "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
            "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
            # ===== Locomotion稳定性奖励（保持但降低权重） =====
            "orientation": -0.05,           # 姿态稳定（降低权重）
            "lin_vel_z": -0.5,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques": -1e-5,               # 扭矩惩罚
            "dof_vel": -5e-5,               # 关节速度惩罚
            "dof_acc": -2.5e-7,             # 关节加速度惩罚
            "action_rate": -0.01,           # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -200.0,          # 终止惩罚
        }
    )

#开始配置类注册装饰器针对不同的导航场景 但先实现基础的平地导航配置，后续在此基础上进行继承和修改以适应不同地形（楼梯、三段地形等）
@registry.envcfg("vbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    max_episode_steps: int = 1000
    sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("vbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """VBot在楼梯地形上的导航配置，继承flat配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("MotrixArena_S1_section001_56")
@dataclass
class VBotSection001EnvCfg(VBotEnvCfg):
    """VBot Section001圆形竞技场导航配置 - 外环出生，中心目标"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section001.xml"
    max_episode_seconds: float = 40.0  # 缩短到0.0秒，配合更多並行环境加速训练
    max_episode_steps: int = 4000  # 缩短到4000步
    render_spacing: float = 0.0    # 修改渲染间距为0，使地图重叠
    
    @dataclass
    class InitState:
        # 圆形场地中心作为参考原点
        pos = [0.0, 0.0, 0.5]

        # 外环出生区域半径范围（两根白线之间）
        # 根据实际场地XML调整这两个值
        spawn_ring_radius_min: float = 10.5 # 内白线半径 [m]
        spawn_ring_radius_max: float = 11.5  # 外白线半径 [m]


        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }

    @dataclass
    class Commands:
        # 目标点固定在场地中心 (0, 0)，yaw随机（朝向不限）
        # 格式: [x_min, y_min, yaw_min, x_max, y_max, yaw_max]
        # x/y 相同则为固定目标；这里目标是绝对坐标(0,0)，在reset中直接赋值
        pose_command_range = [0.0, 0.0, 0, 0.0, 0.0, 0]
        
        # 目标中心位置（蓝色光环中心）
        target_center = [0.0, 0.0, 0.5]
        # 目标位置范围（蓝色光环半径）
        target_range = [0.5, 0.5, 0.0]

    @dataclass
    class ControlConfig:
        action_scale = 0.25

    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("MotrixArena_S1_section01_56")
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """VBot Section01调试配置 - 便于修改和view坐标"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    max_episode_seconds: float = 80.0
    max_episode_steps: int = 8000
    render_spacing: float = 0.0
    
    @dataclass
    class Asset:
        body_name = "base"
        foot_names = ["FR", "FL", "RR", "RL"]
        terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
        ground_subtree = "C"
    
    asset: Asset = field(default_factory=Asset)
    
    @dataclass
    class InitState:
        pos = [0, -2.4, 0.5]  # START区域
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]
        default_joint_angles = {
            "FR_hip_joint": -0.0, "FR_thigh_joint": 0.9, "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.9, "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0, "RR_thigh_joint": 0.9, "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        pose_command_range = [-4, 32.0, 0, 4.0, 32.0, 0]  # 全局终点Y=32
    
    @dataclass
    class ControlConfig:
        action_scale = 0.4

    @dataclass
    class TaskConfig:
        task_name: str = "section01_full"
        enable_landmark_rewards: bool = True
        enable_celebration_reward: bool = True

        # Section011笑脸（3个）
        smile_positions: list[list[float]] = field(
            default_factory=lambda: [[-3.0, 0.1], [0.0, 0.1], [3.0, 0.1]]
        )
        smile_radius: float = 1.3
        smile_reward: float = 10.0

        # 所有红包（共10个：section011的3个 + section012吊桥的1个拜年 + section013河床的6个）
        package_positions: list[list[float]] = field(
            default_factory=lambda: [
                # Section011红包（3个）
                [-3.0, 4.1], [0.0, 4.1], [3.0, 4.1],
                # Section012吊桥拜年红包（1个）
                [-3.0, 18.1],
                # Section013河床红包（6个：5个普通+1个拜年）
                [0.5, 19.5], [-2.9, 18.1], [0.5, 16.0], 
                [2.0, 18.0], [3.5, 16.0], [3.5, 19.5]
            ]
        )
        package_radius: float = 0.8
        package_reward: float = 5.0

        # 里程碑平台（3个需要庆祝的位置）
        milestone_positions: list[list[float]] = field(
            default_factory=lambda: [
                [-3.0, 8.0],   # 2026平台
                [0.0, 24.0],  # 丙午大吉
                [0.0, 32.0]   # 最终终点
            ]
        )
        milestone_reward: float = 50.0  # 每个里程碑奖励

        goal_y: float = 32.0  # 最终终点
        goal_reached_reward: float = 100.0
        celebration_reward: float = 15.0  # 每次庆祝奖励（提高，强化在平台完成动作的动机）
        required_jumps: int = 3  # 每次庆祝需要旋转3圈

        boundary_x: float = 6.0
        boundary_y_max: float = 35.0  # 扩展到终点后
        tilt_threshold_deg: float = 70.0

    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)

@registry.envcfg("MotrixArena_S1_section011_56")
@dataclass
class VBotSection011Cfg(VBotSection01EnvCfg):
    """
    阶段1: Section011 - 上坡带笑脸场景
    从START平台 → 收集笑脸+红包 → 到达2026平台
    """
    max_episode_seconds: float = 60.0
    max_episode_steps: int = 6000
    
    @dataclass
    class InitState:
        pos = [0, -2.4, 0.5]  # START区域
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]
        default_joint_angles = {
            "FR_hip_joint": -0.0, "FR_thigh_joint": 0.9, "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.9, "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0, "RR_thigh_joint": 0.9, "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        pose_command_range = [-2, 8.0, 0, 2.0, 8.0, 0]
    @dataclass
    class ControlConfig:
        action_scale = 0.35  # v7.6: 0.5→0.35，进一步降低首步翻倒风险

    @dataclass
    class TaskConfig:
        task_name: str = "section011"
        enable_landmark_rewards: bool = True
        enable_celebration_reward: bool = True

        smile_positions: list[list[float]] = field(
            default_factory=lambda: [[-3.0, 0.1], [0.0, 0.1], [3.0, 0.1]]
        )
        smile_radius: float = 1.0
        smile_reward: float = 10.0  # 提高到10分（不是比赛的4分）

        package_positions: list[list[float]] = field(
            default_factory=lambda: [[-3.0, 4.1], [0.0, 4.1], [3.0, 4.1]]
        )
        package_radius: float = 0.8
        package_reward: float = 5.0  # 提高到5分（不是比赛的2分）

        goal_y: float = 8.0  # 2026平台位置
        goal_reached_reward: float = 50.0  # 完整收集到达：50分
        celebration_reward: float = 5.0  # 庆祝动作：5分
        required_jumps: int = 3  # 庆祝动作所需"旋转圈数"（3圈=转3×360°）

        boundary_x: float = 6.0
        boundary_y_max: float = 11.0
        tilt_threshold_deg: float = 70.0  # v7.1: 60→70°, 坑洼中40-55°倾斜正常，>70°才算真翻倒
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)

@registry.envcfg("MotrixArena_S1_section012_56")
@dataclass
class VBotSection012Cfg(VBotSection01EnvCfg):
    """
    阶段2: Section012 - 楼梯和吊桥场景
    从2026平台 → 通过楼梯+吊桥/河床 → 到达丙午大吉平台
    """
    max_episode_seconds: float = 50.0
    max_episode_steps: int = 5000
    
    @dataclass
    class InitState:
        pos = [0.0, 8.0, 2.0]  # 2026平台附近
        pos_randomization_range = [-0.2, -0.2, 0.2, 0.2]
        
        default_joint_angles = {
            "FR_hip_joint": -0.0, "FR_thigh_joint": 0.9, "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.9, "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0, "RR_thigh_joint": 0.9, "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标: 丙午大吉平台 (y=24)
        pose_command_range = [-2.0, 24.0, 0.0, 2.0, 24.0, 0.0]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25

    @dataclass
    class TaskConfig:
        task_name: str = "section012"
        enable_landmark_rewards: bool = False
        enable_celebration_reward: bool = False

        smile_positions: list[list[float]] = field(default_factory=list)
        smile_radius: float = 1.0
        smile_reward: float = 4.0

        package_positions: list[list[float]] = field(default_factory=list)
        package_radius: float = 0.8
        package_reward: float = 2.0

        goal_y: float = 24.0
        goal_reached_reward: float = 20.0
        celebration_reward: float = 2.0
        required_jumps: int = 3

        boundary_x: float = 10.0
        boundary_y_max: float = 26.0
        tilt_threshold_deg: float = 60.0
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)

@registry.envcfg("MotrixArena_S1_section013_56")
@dataclass
class VBotSection013Cfg(VBotSection01EnvCfg):
    """
    阶段3: Section013 - 复杂地形越障场景
    从丙午大吉平台 → 滚球区+不规则地形 → 到达中国结平台
    """
    max_episode_seconds: float = 40.0
    max_episode_steps: int = 4000
    
    @dataclass
    class InitState:
        pos = [0.0, 24.0, 2.3]  # 丙午大吉平台附近
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]
        
        default_joint_angles = {
            "FR_hip_joint": -0.0, "FR_thigh_joint": 0.9, "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.9, "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0, "RR_thigh_joint": 0.9, "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标: 中国结平台 (y=32)
        pose_command_range = [-2.0, 32.0, 0.0, 2.0, 32.0, 0.0]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25

    @dataclass
    class TaskConfig:
        task_name: str = "section013"
        enable_landmark_rewards: bool = False
        enable_celebration_reward: bool = False

        smile_positions: list[list[float]] = field(default_factory=list)
        smile_radius: float = 1.0
        smile_reward: float = 4.0

        package_positions: list[list[float]] = field(default_factory=list)
        package_radius: float = 0.8
        package_reward: float = 2.0

        goal_y: float = 32.0
        goal_reached_reward: float = 20.0
        celebration_reward: float = 2.0
        required_jumps: int = 3

        boundary_x: float = 10.0
        boundary_y_max: float = 36.0
        tilt_threshold_deg: float = 60.0
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
