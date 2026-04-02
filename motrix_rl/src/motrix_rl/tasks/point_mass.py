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
from motrix_rl.skrl.config import SkrlCfg


def _configure_point_mass_runner(runner) -> None:
    models = runner.models
    agent = runner.agent
    trainer = runner.trainer

    models.separate = True
    models.policy.hiddens = [32, 32]
    models.value.hiddens = [32, 32]

    agent.rollouts = 16
    agent.learning_epochs = 4
    agent.mini_batches = 4

    trainer.timesteps = 20000


@rlcfg("point_mass", backend="jax")
@dataclass
class PointMassSkrlPpoJax(SkrlCfg):
    """Point mass SKRL configuration for the JAX backend."""

    def __post_init__(self):
        self.num_envs = 256
        runner = self.runner
        agent = runner.agent

        _configure_point_mass_runner(runner)
        agent.learning_rate = 1e-3
        agent.entropy_loss_scale = 0.01
        agent.rewards_shaper_scale = 0.05
        agent.grad_norm_clip = 0.1
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.value_loss_scale = 0.5
        agent.value_clip = 10.0
        agent.clip_predicted_values = False
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.02}
        agent.kl_threshold = 0.03


@rlcfg("point_mass", backend="torch")
@dataclass
class PointMassSkrlPpoTorch(SkrlCfg):
    """Point mass SKRL configuration for the PyTorch backend."""

    def __post_init__(self):
        runner = self.runner
        agent = runner.agent

        models = runner.models

        models.separate = False
        models.policy.hiddens = [32, 32]
        models.value.hiddens = [32, 32]

        agent.rollouts = 32
        agent.learning_epochs = 5
        agent.mini_batches = 4
        runner.trainer.timesteps = 3500
        agent.learning_rate = 3e-4
        agent.entropy_loss_scale = 0.1
