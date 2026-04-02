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
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPPO(SkrlCfg):
        """Anymal C Navigation SKRL PPO configuration.

        Configuration for training ANYmal C robot for flat terrain navigation.
        Uses medium-sized network suitable for most locomotion tasks.
        """

        def __post_init__(self):
            """Configure nested SKRL runner settings."""
            self.num_envs = 2048
            self.play_num_envs = 16
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            # ===== Basic Training Parameters =====
            runner.seed = 42

            # ===== Network Architecture =====
            # Medium-sized network (default configuration, suitable for most tasks)
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            # ===== PPO Core Parameters =====
            agent.rollouts = 48
            agent.learning_epochs = 6
            agent.mini_batches = 32
            agent.learning_rate = 3e-4
            agent.discount_factor = 0.99
            agent.lam = 0.95
            agent.grad_norm_clip = 1.0

            # ===== PPO Clipping Parameters =====
            agent.ratio_clip = 0.2
            agent.value_clip = 0.2
            agent.clip_predicted_values = True

            # ===== Training Parameters =====
            trainer.timesteps = 48000


class rslrl:
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPpoRslrl(RslrlCfg):
        """Anymal C Navigation RSLRL PPO configuration."""

        def __post_init__(self):
            """Configure RSLRL runner and algorithm settings."""
            self.num_envs = 2048
            self.play_num_envs = 16
            runner = self.runner
            algo = runner.algorithm

            # ===== Basic Training Parameters =====
            runner.seed = 42
            # max_iterations = max_env_steps / num_envs / roll_out = 100000000 / 2048 / 48 ≈ 1017
            runner.max_iterations = 1017
            runner.num_steps_per_env = 48
            runner.experiment_name = "anymal_c_navigation_flat"

            # ===== Network Architecture =====
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]

            # ===== Algorithm Parameters =====
            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 6
            algo.num_mini_batches = 4
