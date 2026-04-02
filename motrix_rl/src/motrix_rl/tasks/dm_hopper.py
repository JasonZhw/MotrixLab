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


@rlcfg("dm-hopper-stand", backend="jax")
@dataclass
class HopperStandSkrlJaxPpo(SkrlCfg):
    """Hopper Stand SKRL JAX configuration with nested structure."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Configure model architectures
        models.policy.hiddens = [32, 32, 32]
        models.value.hiddens = [32, 32, 32]

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 4
        agent.learning_rate = 2e-4

        # Configure training parameters
        trainer.timesteps = 20000


@rlcfg("dm-hopper-stand", backend="torch")
@rlcfg("dm-hopper-hop", backend="torch")
@rlcfg("dm-hopper-hop", backend="jax")
@dataclass
class HopperSkrlTorchPpo(SkrlCfg):
    """Hopper SKRL Torch configuration (shared for stand and hop tasks)."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Configure model architectures
        models.policy.hiddens = [32, 32, 32]
        models.value.hiddens = [32, 32, 32]

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 5
        agent.mini_batches = 32
        agent.learning_rate = 2e-4

        # Configure training parameters
        trainer.timesteps = 20000


# ==============================================================================
#  RSLRL Configurations
# ==============================================================================


@rlcfg("dm-hopper-stand")
@dataclass
class HopperStandRslrlPpo(RslrlCfg):
    """Hopper Stand RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 833
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_hopper_stand"
        runner.actor.hidden_dims = [32, 32, 32]
        runner.critic.hidden_dims = [32, 32, 32]
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 4
        algo.num_mini_batches = 4


@rlcfg("dm-hopper-hop")
@dataclass
class HopperHopRslrlPpo(RslrlCfg):
    """Hopper Hop RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 833
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_hopper_hop"
        runner.actor.hidden_dims = [32, 32, 32]
        runner.critic.hidden_dims = [32, 32, 32]
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 5
        algo.num_mini_batches = 32
