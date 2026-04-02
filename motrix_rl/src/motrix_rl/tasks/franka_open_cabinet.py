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
    @rlcfg("franka-open-cabinet")
    @dataclass
    class FrankaOpenCabinetPPO(SkrlCfg):
        """Franka open cabinet - SKRL PPO configuration."""

        def __post_init__(self):
            """Configure nested SKRL runner settings."""
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            # Configure model architectures
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            # Configure PPO agent parameters
            agent.rollouts = 16
            agent.learning_epochs = 5
            agent.mini_batches = 32
            agent.learning_rate = 3e-4
            agent.entropy_loss_scale = 0.001
            agent.rewards_shaper_scale = 1e-1

            # Configure training parameters
            runner.seed = 64
            trainer.timesteps = 24000


class rslrl:
    @rlcfg("franka-open-cabinet")
    @dataclass
    class FrankaOpenCabinetRslrlPpo(RslrlCfg):
        """Franka open cabinet - RSLRL PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            # Runner settings
            runner.seed = 64
            runner.max_iterations = 1500
            runner.num_steps_per_env = 16
            runner.experiment_name = "franka_open_cabinet"

            # Network architecture
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]

            # Algorithm parameters
            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 32
            algo.entropy_coef = 0.001
