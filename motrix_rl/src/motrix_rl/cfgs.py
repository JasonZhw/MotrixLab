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

    @rlcfg("VBotStairsMultiTarget-v0")
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

    @rlcfg("vbot_navigation_stairs_obstacles")
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
        seed: int = 42
        # num_envs: int = 2048
        num_envs: int = 16             
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000
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

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("MotrixArena_S1_section011_56")
    @dataclass
    class VBotNavigationSection011PPOConfig(PPOCfg):
        """
        Section011 - 上坡带笑脸场景 (基础难度)
        从START随机生成 → 收集笑脸/红包 → 到达2026平台做庆祝
        策略: 全力训练，最大学习率用于从零开始学习
        """
        seed: int = 42
        num_envs: int = 8192              # 最大并行度，加速初期学习
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000  # ~7500个训练迭代
        check_point_interval: int = 500

        learning_rate: float = 5e-4       # 最高学习率，从零开始
        rollouts: int = 32                # 保持batch大小: 8192 * 32 = 262K samples/iter
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

    @rlcfg("MotrixArena_S1_section012_56")
    @dataclass
    class VBotNavigationSection012PPOConfig(PPOCfg):
        """
        Section012 - 楼梯和吊桥场景 (中等难度)
        从2026随机生成 → 波浪+楼梯+吊桥/河床 → 收集红包 → 丙午大吉庆祝
        策略: 迁移学习，降低学习率2.5倍用于细调
        加载section011的checkpoint继续训练
        """
        seed: int = 42
        num_envs: int = 8192
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000  # 同等训练预算
        check_point_interval: int = 500

        learning_rate: float = 2e-4       # 2.5倍降低，细调已学特征
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
    class VBotNavigationSection013PPOConfig(PPOCfg):
        """
        Section013 - 复杂地形越障场景 (高等难度)
        从丙午大吉随机生成 → 滚球碰撞恢复 → 不规则地形 → 中国结平台庆祝
        策略: 迁移学习，降低学习率5倍用于精调
        加载section012的checkpoint继续训练
        """
        seed: int = 42
        num_envs: int = 8192
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000  # 同等训练预算
        check_point_interval: int = 500

        learning_rate: float = 1e-4       # 5倍降低，精调碰撞恢复策略
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