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
    @rlcfg("rm65-open-cabinet", "torch")
    @dataclass
    class RM65OpenCabinetPPOTorch(SkrlCfg):
        """RM65 open cabinet - SKRL Torch PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            models.separate = True
            models.policy.hiddens = [512, 256, 128]
            models.value.hiddens = [512, 256, 128]

            agent.rollouts = 32
            agent.learning_epochs = 5
            agent.mini_batches = 32
            agent.learning_rate = 8e-5
            agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.005}
            agent.ratio_clip = 0.10
            agent.grad_norm_clip = 0.4
            agent.entropy_loss_scale = 2e-4
            agent.rewards_shaper_scale = 5e-2

            runner.seed = 64
            trainer.timesteps = 21000


class rslrl:
    @rlcfg("rm65-open-cabinet")
    @dataclass
    class RM65OpenCabinetRslrlPpo(RslrlCfg):
        """RM65 open cabinet - RSLRL PPO configuration."""

        def __post_init__(self):
            runner = self.runner
            algo = runner.algorithm

            runner.seed = 64
            runner.max_iterations = 1500
            runner.num_steps_per_env = 24
            runner.experiment_name = "rm65_open_cabinet"

            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]

            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 8
            algo.entropy_coef = 0.001
