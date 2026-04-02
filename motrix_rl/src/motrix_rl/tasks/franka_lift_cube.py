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
    @rlcfg("franka-lift-cube", "jax")
    @dataclass
    class FrankaLiftPPOJax(SkrlCfg):
        """Franka lift cube - SKRL JAX PPO configuration."""

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
            agent.mini_batches = 32
            agent.learning_rate = 1e-3

            # Configure training parameters
            trainer.timesteps = 100000

    @rlcfg("franka-lift-cube", "torch")
    @dataclass
    class FrankaLiftPPOTorch(SkrlCfg):
        """Franka lift cube - SKRL Torch PPO configuration."""

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
            agent.learning_epochs = 8
            agent.mini_batches = 4
            agent.learning_rate = 3e-4
            agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
            agent.entropy_loss_scale = 0.001
            agent.rewards_shaper_scale = 0.01

            # Configure training parameters
            trainer.timesteps = 100000


class rslrl:
    @rlcfg("franka-lift-cube")
    @dataclass
    class FrankaLiftRslrlPpo(RslrlCfg):
        """Franka lift cube - RSLRL PPO configuration."""

        def __post_init__(self):
            """Configure RSLRL runner and algorithm settings."""
            runner = self.runner
            algo = runner.algorithm

            # Runner settings
            runner.seed = 42
            runner.max_iterations = 500
            runner.num_steps_per_env = 64
            runner.experiment_name = "franka_lift_cube"

            # Network architecture
            runner.actor.hidden_dims = [256, 128, 128]
            runner.critic.hidden_dims = [256, 128, 128]

            # Algorithm parameters (match SKRL Torch config)
            algo.learning_rate = 5e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 4
            algo.entropy_coef = 1e-3
