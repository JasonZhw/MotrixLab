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

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class basic:
    @rlcfg("cartpole")
    @dataclass
    class CartPolePPO(PPOCfg):
        max_env_steps: int = 10_000_000
        check_point_interval: int = 500

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 4

    @rlcfg("bounce_ball")
    @dataclass
    class BounceBallPPO(PPOCfg):
        max_env_steps: int = 50_000_000
        check_point_interval: int = 5000

        # Override PPO configuration for bounce ball task
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 512, 512)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 512, 512)
        rollouts: int = 128
        learning_epochs: int = 15
        mini_batches: int = 16
        learning_rate: float = 2e-4
        num_envs: int = 1024

    @rlcfg("dm-walker", backend="jax")
    @rlcfg("dm-stander", backend="jax")
    @rlcfg("dm-runner", backend="jax")
    @dataclass
    class WalkerPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 4

    @rlcfg("dm-stander", backend="torch")
    @rlcfg("dm-walker", backend="torch")
    @dataclass
    class WalkerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32

    @rlcfg("dm-runner", backend="torch")
    @dataclass
    class RunnerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 2
        mini_batches: int = 32

    @rlcfg("dm-cheetah", backend="jax")
    @dataclass
    class CheetahPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("dm-cheetah", backend="torch")
    @dataclass
    class CheetahPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("dm-hopper-stand", backend="jax")
    @rlcfg("dm-hopper-hop", backend="jax")
    @dataclass
    class HopperPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 5
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-hopper-stand", backend="torch")
    @rlcfg("dm-hopper-hop", backend="torch")
    @dataclass
    class HopperPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 5
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-reacher", backend="jax")
    @dataclass
    class ReacherPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-reacher", backend="torch")
    @dataclass
    class ReacherPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)


class locomotion:
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkPPO(PPOCfg):
        """
        Go1 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("go2-flat-terrain-walk")
    @dataclass
    class Go2WalkPPO(PPOCfg):
        """
        Go2 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("go1-rough-terrain-walk")
    @dataclass
    class Go1WalkRoughPPO(Go1WalkPPO):
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("go1-stairs-terrain-walk")
    @dataclass
    class Go1WalkStairsPPO(Go1WalkRoughPPO): ...


class manipulation:
    @rlcfg("franka-lift-cube")
    @dataclass
    class FrankaLiftPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 4096 * 50000
        check_point_interval: int = 500
        share_policy_value_features: bool = True

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 4
        learning_rate: float = 3e-4
        learning_rate_scheduler_kl_threshold: float = 0.01
        entropy_loss_scale: float = 0.001
        rewards_shaper_scale: float = 0.01

    @rlcfg("franka-open-cabinet")
    @dataclass
    class FrankaOpenCabinetPPO(PPOCfg):
        seed: int = 64
        max_env_steps: int = 2048 * 24000
        check_point_interval: int = 500
        share_policy_value_features: bool = False

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 16
        learning_epochs: int = 5
        mini_batches: int = 32
        learning_rate: float = 3e-4
        rewards_shaper_scale: float = 1e-1
        entropy_loss_scale: float = 0.001


class navigation:
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPPOConfig(PPOCfg):
        # ===== Basic Training Parameters =====
        seed: int = 42  # Random seed
        num_envs: int = 2048  # Number of parallel environments during training
        play_num_envs: int = 16  # Number of parallel environments during evaluation
        max_env_steps: int = 100_000_000  # Maximum training steps
        check_point_interval: int = 1000  # Checkpoint save interval (save every 100 iterations)

        # ===== PPO Algorithm Core Parameters =====
        learning_rate: float = 3e-4  # Learning rate
        rollouts: int = 48  # Number of experience replay rollouts
        learning_epochs: int = 6  # Number of training epochs per update
        mini_batches: int = 32  # Number of mini-batches
        discount_factor: float = 0.99  # Discount factor
        lambda_param: float = 0.95  # GAE parameter
        grad_norm_clip: float = 1.0  # Gradient clipping

        # ===== PPO Clipping Parameters =====
        ratio_clip: float = 0.2  # PPO clipping ratio
        value_clip: float = 0.2  # Value clipping
        clip_predicted_values: bool = True  # Clip predicted values

        # Medium-sized network (default configuration, suitable for most tasks)
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_flat")
    @dataclass
    class VBotNavigationPPOConfig(PPOCfg):
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_stairs")
    @dataclass
    class VBotNavigationStairsPPOConfig(PPOCfg):
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000  
        num_envs: int = 2048

       
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("MotrixArena_S1_section001_56")
    @dataclass
    class VBotNavigationSection001PPOConfig(PPOCfg):

        seed: int = 42
        num_envs: int = 2048  # 从4096扩大到8192，并行度翻倍加速训练
        play_num_envs: int = 10
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 500

        learning_rate: float = 3e-4  
        rollouts: int = 48  
        learning_epochs: int = 6
        mini_batches: int = 32  
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)



    @rlcfg("MotrixArena_S1_section01_56")
    @dataclass
    class VBotNavigationSection01PPOConfig(PPOCfg):
        """完整Section01训练配置 (最终验证用)"""
        seed: int = 42
        num_envs: int = 2048  
        play_num_envs: int = 1
        max_env_steps: int = 2048 * 70_000
        check_point_interval: int = 1000
        share_policy_value_features: bool =True # 共享特征：全局策略统一表征，减少参数量加速收敛

        learning_rate: float = 5e-4
        rollouts: int = 64
        learning_epochs: int = 6
        mini_batches: int = 64  # batch=4096×32=131072, mini_batch=2048
        discount_factor: float = 0.995  # 长路线/楼梯任务更重视长期回报
        lambda_param: float = 0.97
        grad_norm_clip: float = 1.0
        entropy_loss_scale: float = 0.008  # 保留适度探索，降低楼梯卡住局部最优概率

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        # obs=67维, act=12维, 共享特征提取
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("MotrixArena_S1_section011_56")
    @dataclass
    class VBotNavigationSection01Phase1PPO(PPOCfg):
        """阶段1: Section011 - 上坡笑脸场景 (v7.1)"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 500

        learning_rate: float = 5e-4
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 64
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        entropy_loss_scale: float = 0.01  # 鼓励探索，解决不进洼地的关键

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("MotrixArena_S1_section012_56")
    @dataclass
    class VBotNavigationSection01Phase2PPO(PPOCfg):
        """阶段2: Section012 - 楼梯吊桥场景 (加载Phase1 checkpoint)"""
        seed: int = 42
        num_envs: int = 2048    
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 500

        learning_rate: float = 2e-4  # 降低2.5倍
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 64
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("MotrixArena_S1_section013_56")
    @dataclass
    class VBotNavigationSection01Phase3PPO(PPOCfg):
        """阶段3: Section013 - 碰撞恢复场景 (加载Phase2 checkpoint)"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 500

        learning_rate: float = 1e-4  # 降低5倍
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 64
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)