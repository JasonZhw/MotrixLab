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
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.skrl.config import SkrlCfg


class skrl:
    @rlcfg("shadow-hand-repose")
    @dataclass
    class ShadowHandReposePPO(SkrlCfg):
        """Shadow Hand Repose PPO configuration.

        Configuration for training Shadow Hand to reach target hand pose.
        Uses large-scale parallel training with 8192 environments.
        """

        def __post_init__(self):
            """Configure nested SKRL runner settings."""
            self.num_envs = 8192
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            # ===== Basic Settings =====
            runner.seed = 42

            # ===== Network Architecture =====
            models.policy.hiddens = [512, 512, 256, 128]
            models.value.hiddens = [512, 512, 256, 128]

            # ===== PPO Core Parameters =====
            agent.rollouts = 16
            agent.learning_epochs = 5
            agent.mini_batches = 4
            agent.discount_factor = 0.99
            agent.lam = 0.95

            # ===== Learning Rate =====
            agent.learning_rate = 5.0e-04
            agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.016}

            # ===== Clipping =====
            agent.ratio_clip = 0.2
            agent.value_clip = 0.2
            agent.clip_predicted_values = True
            agent.grad_norm_clip = 1.0

            # ===== Loss Coefficients =====
            agent.entropy_loss_scale = 0.0
            agent.value_loss_scale = 2.0
            agent.kl_threshold = 0.0

            # ===== Reward Shaping =====
            agent.rewards_shaper_scale = 0.01

            # ===== Training Control =====
            agent.random_timesteps = 0
            agent.learning_starts = 0
            agent.time_limit_bootstrap = False

            # ===== Training Parameters =====
            trainer.timesteps = 24000


class rslrl:
    @rlcfg("shadow-hand-repose")
    @dataclass
    class ShadowHandReposeRslrlPpo(RslrlCfg):
        """Shadow Hand Repose RSLRL PPO configuration."""

        def __post_init__(self):
            """Configure RSLRL runner and algorithm settings."""
            self.num_envs = 8192
            runner = self.runner
            algo = runner.algorithm

            # ===== Basic Settings =====
            runner.seed = 42

            # max_iterations = max_env_steps / num_envs / roll_out = 200000000 / 8192 / 16 ≈ 1525
            runner.max_iterations = 1500
            runner.num_steps_per_env = 16
            runner.experiment_name = "shadow_hand_repose"

            # ===== Network Architecture =====
            runner.actor.hidden_dims = [512, 512, 256, 128]
            runner.critic.hidden_dims = [512, 512, 256, 128]

            # ===== Algorithm Parameters =====
            algo.learning_rate = 5.0e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 0.0
