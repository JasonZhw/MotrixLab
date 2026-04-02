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
    @rlcfg("dm-quadruped-walk", backend="jax")
    @rlcfg("dm-quadruped-run", backend="jax")
    @rlcfg("dm-quadruped-escape", backend="jax")
    @rlcfg("dm-quadruped-fetch", backend="jax")
    @dataclass
    class QuadrupedSkrlPpoJax(SkrlCfg):
        """DM quadruped tasks - SKRL PPO configuration (JAX)."""

        def __post_init__(self):
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            models.policy.clip_actions = False
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            agent.rollouts = 24
            agent.learning_epochs = 4
            agent.mini_batches = 32
            agent.learning_rate = 2e-4

            trainer.timesteps = 20000

    @rlcfg("dm-quadruped-walk", backend="torch")
    @rlcfg("dm-quadruped-run", backend="torch")
    @rlcfg("dm-quadruped-escape", backend="torch")
    @rlcfg("dm-quadruped-fetch", backend="torch")
    @dataclass
    class QuadrupedSkrlPpoTorch(SkrlCfg):
        """DM quadruped tasks - SKRL PPO configuration (PyTorch)."""

        def __post_init__(self):
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            models.policy.clip_actions = False
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            agent.rollouts = 24
            agent.learning_epochs = 4
            agent.mini_batches = 32
            agent.learning_rate = 2e-4

            trainer.timesteps = 27000


class rslrl:
    @rlcfg("dm-quadruped-walk")
    @rlcfg("dm-quadruped-run")
    @rlcfg("dm-quadruped-escape")
    @dataclass
    class QuadrupedRslrlPpo(RslrlCfg):
        """DM quadruped tasks - RSLRL PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm
            runner.seed = 42
            runner.max_iterations = 1667
            runner.num_steps_per_env = 24
            runner.experiment_name = "dm_quadruped"
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]
            algo.learning_rate = 2e-4
            algo.num_learning_epochs = 4
            algo.num_mini_batches = 32
