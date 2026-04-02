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


@rlcfg("dm-finger-spin")
@rlcfg("dm-finger-turn-easy")
@rlcfg("dm-finger-turn-hard")
@dataclass
class FingerSkrlPpo(SkrlCfg):
    """Finger SKRL configuration with nested structure."""

    def __post_init__(self):
        """Configure nested SKRL runner settings."""
        runner = self.runner
        agent = runner.agent
        trainer = runner.trainer

        # Configure PPO agent parameters
        agent.rollouts = 24
        agent.learning_epochs = 4
        agent.mini_batches = 4
        agent.learning_rate = 2e-4

        # Configure training parameters
        trainer.timesteps = 20000


# @rlcfg("dm-finger-spin", backend="jax")
# @dataclass
# class FingerSpinSkrlPpoJax(SkrlCfg):
#     """Finger Spin SKRL configuration for JAX backend.

#     More conservative PPO for stability (spin can collapse mid-training in JAX).
#     """

#     def __post_init__(self):
#         """Configure nested SKRL runner settings."""
#         runner = self.runner
#         agent = runner.agent
#         trainer = runner.trainer

#         # Configure PPO agent parameters (conservative for stability)
#         agent.rollouts = 24
#         agent.learning_epochs = 1
#         agent.mini_batches = 16
#         agent.learning_rate = 7.5e-5
#         agent.ratio_clip = 0.08
#         agent.value_clip = 0.1
#         agent.value_loss_scale = 0.5
#         agent.grad_norm_clip = 0.25
#         agent.entropy_loss_scale = 5e-4
#         agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.003}

#         # Configure training parameters
#         trainer.timesteps = 20000


# @rlcfg("dm-finger-turn-hard", backend="jax")
# @dataclass
# class FingerTurnHardSkrlPpoJax(SkrlCfg):
#     """Finger Turn Hard SKRL configuration for JAX backend.

#     Extra conservative to avoid late-stage collapses.
#     """

#     def __post_init__(self):
#         """Configure nested SKRL runner settings."""
#         runner = self.runner
#         agent = runner.agent
#         trainer = runner.trainer

#         # Configure PPO agent parameters (extra conservative)
#         agent.rollouts = 24
#         agent.learning_epochs = 1
#         agent.mini_batches = 16
#         agent.learning_rate = 5e-5
#         agent.ratio_clip = 0.08
#         agent.value_loss_scale = 0.5
#         agent.grad_norm_clip = 0.25
#         agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.004}

#         # Configure training parameters
#         trainer.timesteps = 20000


@rlcfg("dm-finger-spin")
@rlcfg("dm-finger-turn-easy")
@rlcfg("dm-finger-turn-hard")
@dataclass
class FingerRslrlPpo(RslrlCfg):
    """Finger RSLRL configuration."""

    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm
        runner.seed = 42
        runner.max_iterations = 500
        runner.num_steps_per_env = 24
        runner.experiment_name = "dm_finger"
        algo.learning_rate = 2e-4
        algo.num_learning_epochs = 4
        algo.num_mini_batches = 4
