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

"""Configuration for dm-walker, dm-stander, and dm-runner tasks.

These three tasks share similar configurations:
- JAX: dm-walker, dm-stander, dm-runner all use the same config
- Torch: dm-walker and dm-stander share one config, dm-runner has a different config
- RSLRL: Each task has its own config
"""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.skrl.config import SkrlCfg

# =============================================================================
# SKRL JAX Configurations
# =============================================================================


@rlcfg("dm-walker", backend="jax")
@rlcfg("dm-stander", backend="jax")
@rlcfg("dm-runner", backend="jax")
@dataclass
class DmRunnerSkrlJaxCfg(SkrlCfg):
    """Shared SKRL JAX configuration for dm-walker, dm-stander, and dm-runner."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        runner.seed = 42

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 4
        agent.learning_rate = 2e-4

        # Configure training parameters
        # trainer.timesteps = max_env_steps / num_envs
        runner.trainer.timesteps = 20000


# =============================================================================
# SKRL Torch Configurations
# =============================================================================


@rlcfg("dm-walker", backend="torch")
@rlcfg("dm-stander", backend="torch")
@dataclass
class DmWalkerStanderSkrlTorchCfg(SkrlCfg):
    """Shared SKRL Torch configuration for dm-walker and dm-stander."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        runner.seed = 42

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 32
        agent.learning_rate = 2e-4

        # Configure training parameters
        runner.trainer.timesteps = 20000


@rlcfg("dm-runner", backend="torch")
@dataclass
class DmRunnerSkrlTorchCfg(SkrlCfg):
    """dm-runner SKRL Torch configuration with nested structure."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        runner.seed = 42

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 24
        agent.learning_epochs = 2
        agent.mini_batches = 32
        agent.learning_rate = 2e-4

        runner.trainer.timesteps = 20000


# =============================================================================
# RSLRL Configurations
# =============================================================================


@rlcfg("dm-walker")
@dataclass
class WalkerRslrlPpo(RslrlCfg):
    """dm-walker RSLRL configuration.

    Note: max_iterations = max_env_steps / num_envs / roll_out
                     = 1024 * 40000 / 2048 / 24 ≈ 833
    """

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 833
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_walker"
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 4
        algo.num_mini_batches = 4


@rlcfg("dm-stander")
@dataclass
class StanderRslrlPpo(RslrlCfg):
    """dm-stander RSLRL configuration.

    Note: max_iterations = max_env_steps / num_envs / roll_out
                     = 1024 * 40000 / 2048 / 24 ≈ 833
    """

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 833
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_stander"
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 4
        algo.num_mini_batches = 32


@rlcfg("dm-runner")
@dataclass
class RunnerRslrlPpo(RslrlCfg):
    """dm-runner RSLRL configuration.

    Note: max_iterations = max_env_steps / num_envs / roll_out
                     = 1024 * 40000 / 2048 / 24 ≈ 833
    """

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 833
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_runner"
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 2
        algo.num_mini_batches = 32
