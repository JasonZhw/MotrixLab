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


@rlcfg("dm-manipulator-bring-ball", backend="jax")
@dataclass
class ManipulatorSkrlPpoJax(SkrlCfg):
    """Manipulator SKRL configuration with nested structure (JAX)."""

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
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 4
        agent.learning_rate = 3e-4
        agent.ratio_clip = 0.2
        agent.entropy_loss_scale = 1e-3
        agent.grad_norm_clip = 1.0

        # Configure training parameters
        trainer.timesteps = 20000


@rlcfg("dm-manipulator-bring-ball", backend="torch")
@dataclass
class ManipulatorSkrlPpoTorch(SkrlCfg):
    """Manipulator SKRL configuration with nested structure (PyTorch)."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Configure model architectures (PyTorch - different network)
        models.policy.hiddens = [256, 256]
        models.value.hiddens = [256, 256]

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 4
        agent.learning_rate = 2e-4

        # Configure training parameters
        trainer.timesteps = 20000


@rlcfg("dm-manipulator-bring-ball")
@dataclass
class ManipulatorRslrlPpo(RslrlCfg):
    """Manipulator RSLRL configuration"""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 500
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_manipulator"
        runner.actor.hidden_dims = [256, 128, 64]
        runner.critic.hidden_dims = [256, 128, 64]
        algo.learning_rate = 3e-4
        algo.num_learning_epochs = 4
        algo.num_mini_batches = 4
