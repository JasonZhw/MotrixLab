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

"""PPO defaults for Microduck flat-terrain velocity training."""

from dataclasses import dataclass, field

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import RslrlCfg, RslrlRunnerCfg
from motrix_rl.skrl.config import SkrlCfg, SkrlRunnerCfg

ENV_NAME = "microduck-flat-terrain-walk"


def _skrl_runner_cfg() -> SkrlRunnerCfg:
    cfg = SkrlRunnerCfg()
    cfg.models.policy.hiddens = [256, 128, 64]
    cfg.models.policy.output_activation = "tanh"
    cfg.models.policy.clip_actions = True
    cfg.models.value.hiddens = [256, 128, 64]
    cfg.agent.rollouts = 24
    cfg.agent.learning_epochs = 5
    cfg.agent.mini_batches = 4
    cfg.agent.learning_rate = 3.0e-4
    cfg.trainer.timesteps = 4_000 * 24 * 4096
    return cfg


def _rslrl_runner_cfg() -> RslrlRunnerCfg:
    cfg = RslrlRunnerCfg()
    cfg.seed = 42
    cfg.max_iterations = 5000
    cfg.num_steps_per_env = 24
    cfg.experiment_name = "microduck_flat_terrain_walk"
    cfg.actor.hidden_dims = [512, 256, 128]
    cfg.actor.obs_normalization = True
    cfg.critic.hidden_dims = [512, 256, 128]
    cfg.critic.obs_normalization = True
    cfg.algorithm.learning_rate = 1.0e-3
    cfg.algorithm.num_learning_epochs = 5
    cfg.algorithm.num_mini_batches = 4
    cfg.algorithm.desired_kl = 0.01
    cfg.algorithm.entropy_coef = 0.01
    return cfg


class skrl:
    @rlcfg(ENV_NAME)
    @dataclass
    class MicroduckWalkSkrlPpo(SkrlCfg):
        num_envs: int = 4096
        play_num_envs: int = 16
        runner: SkrlRunnerCfg = field(default_factory=_skrl_runner_cfg)


class rslrl:
    @rlcfg(ENV_NAME)
    @dataclass
    class MicroduckWalkRslrlPpo(RslrlCfg):
        num_envs: int = 4096
        play_num_envs: int = 16
        runner: RslrlRunnerCfg = field(default_factory=_rslrl_runner_cfg)
