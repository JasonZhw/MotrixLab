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


@rlcfg("dm-humanoid-stand", backend="jax")
@rlcfg("dm-humanoid-walk", backend="jax")
@rlcfg("dm-humanoid-run", backend="jax")
@dataclass
class HumanoidSkrlPpo(SkrlCfg):
    """Humanoid SKRL configuration with nested structure (JAX)."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Configure model architectures
        models.policy.hiddens = [512, 256, 128]
        models.value.hiddens = [512, 256, 128]

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 8
        agent.mini_batches = 2
        agent.learning_rate = 3e-4

        # Configure training parameters
        trainer.timesteps = 20000


@rlcfg("dm-humanoid-stand", backend="torch")
@rlcfg("dm-humanoid-walk", backend="torch")
@rlcfg("dm-humanoid-run", backend="torch")
@dataclass
class HumanoidSkrlPpoTorch(SkrlCfg):
    """Humanoid SKRL configuration with nested structure (PyTorch)."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Configure model architectures
        models.policy.hiddens = [512, 256, 128]
        models.value.hiddens = [512, 256, 128]

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 8
        agent.mini_batches = 2
        agent.learning_rate = 3e-4

        # Configure training parameters
        trainer.timesteps = 20000


@rlcfg("dm-humanoid-stand")
@rlcfg("dm-humanoid-walk")
@rlcfg("dm-humanoid-run")
@dataclass
class HumanoidRslrlPpo(RslrlCfg):
    """Humanoid RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 400
        runner.num_steps_per_env = 64
        runner.experiment_name = "dm_humanoid"
        runner.actor.hidden_dims = [512, 256, 128]
        runner.critic.hidden_dims = [512, 256, 128]
        algo.learning_rate = 3e-4
        algo.num_learning_epochs = 5
        algo.num_mini_batches = 4
        algo.entropy_coef = 0.001
