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


@rlcfg("dm-lqr-2-1", backend="jax")
@dataclass
class Lqr21SkrlJaxPpo(SkrlCfg):
    def __post_init__(self):
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        models.policy.hiddens = [128, 128, 64]
        models.value.hiddens = [128, 128, 64]

        agent.rollouts = 64
        agent.learning_epochs = 8
        agent.mini_batches = 8
        agent.learning_rate = 1.5e-4
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.grad_norm_clip = 0.5
        agent.ratio_clip = 0.12
        agent.value_clip = 0.1
        agent.value_loss_scale = 1.0
        agent.entropy_loss_scale = 1e-3
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}

        trainer.timesteps = 2500


@rlcfg("dm-lqr-2-1", backend="torch")
@dataclass
class Lqr21SkrlTorchPpo(SkrlCfg):
    def __post_init__(self):
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        models.policy.hiddens = [128, 128, 64]
        models.value.hiddens = [128, 128, 64]

        agent.rollouts = 64
        agent.learning_epochs = 8
        agent.mini_batches = 8
        agent.learning_rate = 1.5e-4
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.grad_norm_clip = 0.5
        agent.ratio_clip = 0.12
        agent.value_clip = 0.1
        agent.value_loss_scale = 1.0
        agent.entropy_loss_scale = 1e-3
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}

        trainer.timesteps = 2500


@rlcfg("dm-lqr-6-2", backend="jax")
@dataclass
class Lqr62SkrlJaxPpo(SkrlCfg):
    def __post_init__(self):
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        models.policy.hiddens = [256, 128, 64]
        models.value.hiddens = [256, 128, 64]

        agent.rollouts = 96
        agent.learning_epochs = 8
        agent.mini_batches = 16
        agent.learning_rate = 1.0e-4
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.grad_norm_clip = 0.5
        agent.ratio_clip = 0.12
        agent.value_clip = 0.1
        agent.value_loss_scale = 1.0
        agent.entropy_loss_scale = 1e-3
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}

        trainer.timesteps = 20000


@rlcfg("dm-lqr-6-2", backend="torch")
@dataclass
class Lqr62SkrlTorchPpo(SkrlCfg):
    def __post_init__(self):
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        models.policy.hiddens = [256, 128, 64]
        models.value.hiddens = [256, 128, 64]

        agent.rollouts = 96
        agent.learning_epochs = 8
        agent.mini_batches = 16
        agent.learning_rate = 1.0e-4
        agent.discount_factor = 0.995
        agent.lam = 0.97
        agent.grad_norm_clip = 0.5
        agent.ratio_clip = 0.12
        agent.value_clip = 0.1
        agent.value_loss_scale = 1.0
        agent.entropy_loss_scale = 1e-3
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}

        trainer.timesteps = 20000


@rlcfg("dm-lqr-2-1")
@dataclass
class Lqr21RslrlPpo(RslrlCfg):
    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm

        runner.seed = 42
        runner.max_iterations = 1500
        runner.num_steps_per_env = 64
        runner.experiment_name = "dm_lqr_2_1"
        runner.actor.hidden_dims = [128, 128, 64]
        runner.critic.hidden_dims = [128, 128, 64]

        algo.learning_rate = 1.5e-4
        algo.num_learning_epochs = 8
        algo.num_mini_batches = 8
        algo.gamma = 0.995
        algo.lam = 0.97
        algo.clip_param = 0.12
        algo.desired_kl = 0.01
        algo.entropy_coef = 1e-3
        algo.max_grad_norm = 0.5


@rlcfg("dm-lqr-6-2")
@dataclass
class Lqr62RslrlPpo(RslrlCfg):
    def __post_init__(self):
        runner = self.runner
        algo = runner.algorithm

        runner.seed = 42
        runner.max_iterations = 2000
        runner.num_steps_per_env = 96
        runner.experiment_name = "dm_lqr_6_2"
        runner.actor.hidden_dims = [256, 128, 64]
        runner.critic.hidden_dims = [256, 128, 64]

        algo.learning_rate = 1.0e-4
        algo.num_learning_epochs = 8
        algo.num_mini_batches = 16
        algo.gamma = 0.995
        algo.lam = 0.97
        algo.clip_param = 0.12
        algo.desired_kl = 0.01
        algo.entropy_coef = 1e-3
        algo.max_grad_norm = 0.5
