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


# -- docs-tag-start: acrobot-train-cfg --
@rlcfg("acrobot", backend="jax")
@dataclass
class AcrobotSkrlPpo(SkrlCfg):
    """Acrobot SKRL configuration with nested structure.

    Configuration overrides:
    - Network architecture: 32x32 hidden layers for both policy and value
    - PPO parameters: 64 rollouts, 5 learning epochs, 8 mini-batches
    - Training: 60M timesteps
    """

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner

        # Configure model architectures
        runner.models.policy.hiddens = [32, 32]
        runner.models.value.hiddens = [32, 32]

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 64
        agent.learning_epochs = 5
        agent.mini_batches = 8
        agent.learning_rate = 3e-4
        agent.grad_norm_clip = 0.1
        agent.entropy_loss_scale = 0.1
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.ratio_clip = 0.2
        agent.value_loss_scale = 0.5
        agent.value_clip = 10.0
        agent.clip_predicted_values = False
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.02}
        agent.kl_threshold = 0.03

        # Configure training parameters
        runner.trainer.timesteps = 29000


# -- docs-tag-end: acrobot-train-cfg --


@rlcfg("acrobot", backend="torch")
@dataclass
class AcrobotSkrlPpoTorch(SkrlCfg):
    """Acrobot SKRL Torch configuration with nested structure.

    Configuration overrides:
    - Network architecture: 32x32 hidden layers for both policy and value
    - PPO parameters: 64 rollouts, 5 learning epochs, 8 mini-batches
    - Training: 60M timesteps
    - Torch-specific: entropy_loss_scale=0.2 (vs 0.1 for JAX)
    """

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner

        # Configure model architectures
        runner.models.policy.hiddens = [32, 32]
        runner.models.value.hiddens = [32, 32]

        # Configure PPO agent parameters
        agent = runner.agent
        agent.rollouts = 64
        agent.learning_epochs = 5
        agent.mini_batches = 8
        agent.learning_rate = 3e-4
        agent.grad_norm_clip = 0.1
        agent.entropy_loss_scale = 0.2  # Torch-specific: higher than JAX (0.1)
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.ratio_clip = 0.2
        agent.value_loss_scale = 0.5
        agent.value_clip = 10.0
        agent.clip_predicted_values = False
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.02}
        agent.kl_threshold = 0.03

        # Configure training parameters
        runner.trainer.timesteps = 29000


@rlcfg("acrobot")
@dataclass
class AcrobotRslrlPpo(RslrlCfg):
    """Acrobot RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 900
        runner.num_steps_per_env = 32
        runner.experiment_name = "acrobot"
        runner.actor.hidden_dims = [32, 32]
        runner.critic.hidden_dims = [32, 32]
        algo.learning_rate = 1e-4
        algo.entropy_coef = 0.005
