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


# -- docs-tag-start: pendulum-train-cfg --
@rlcfg("pendulum")
@dataclass
class PendulumSkrlPpo(SkrlCfg):
    """Pendulum SKRL configuration with nested structure."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner

        # Configure model architectures
        runner.models.policy.hiddens = [64, 64]
        runner.models.value.hiddens = [64, 64]

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 32
        agent.learning_epochs = 5
        agent.mini_batches = 4
        agent.learning_rate = 3e-4
        # Configure training parameters
        # trainer.timesteps = max_env_steps / num_envs
        runner.trainer.timesteps = 5000


# -- docs-tag-end: pendulum-train-cfg --


@rlcfg("pendulum")
@dataclass
class PendulumRslrlPpo(RslrlCfg):
    """Pendulum RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 150
        runner.num_steps_per_env = 32
        runner.experiment_name = "pendulum"
        runner.actor.hidden_dims = [64, 64]
        runner.critic.hidden_dims = [64, 64]

        algo.learning_rate = 3e-4
        algo.entropy_coef = 0.005
        algo.num_learning_epochs = 5
        algo.num_mini_batches = 4
