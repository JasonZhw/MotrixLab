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
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkFlatSkrlPpo(SkrlCfg):
        """Go1 robot walk on flat terrain - SKRL PPO configuration."""

        def __post_init__(self):
            """Configure nested SKRL runner settings."""
            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            # Configure model architectures (medium size network)
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            # Configure PPO agent parameters
            agent.rollouts = 24
            agent.learning_epochs = 5
            agent.mini_batches = 3
            agent.learning_rate = 3e-4

            # Configure training parameters
            trainer.timesteps = 30000

    @rlcfg("go1-rough-terrain-walk")
    @dataclass
    class Go1WalkRoughSkrlPpo(Go1WalkFlatSkrlPpo):
        """Go1 robot walk on rough terrain - SKRL PPO configuration.

        Uses larger network than flat terrain for more complex terrain handling.
        """

        def __post_init__(self):
            """Configure nested SKRL runner settings."""
            runner = self.runner
            models = runner.models
            # Configure model architectures (larger network for rough terrain)
            models.policy.hiddens = [512, 256, 128]
            models.value.hiddens = [512, 256, 128]

    @rlcfg("go1-stairs-terrain-walk")
    @dataclass
    class Go1WalkStairsPPO(Go1WalkRoughSkrlPpo): ...


class rslrl:
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkFlatRslrlPpo(RslrlCfg):
        """Go1 robot walk on flat terrain - RSLRL PPO configuration."""

        def __post_init__(self):
            """Configure RSLRL runner and algorithm settings."""
            runner = self.runner
            algo = runner.algorithm

            # Runner settings
            runner.seed = 42
            runner.max_iterations = 1000
            runner.num_steps_per_env = 24
            runner.experiment_name = "go1_flat_terrain_walk"

            # Network architecture (medium size for flat terrain)
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]

            # Algorithm parameters (match SKRL config)
            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 5
            algo.num_mini_batches = 3

    @rlcfg("go1-rough-terrain-walk")
    @dataclass
    class Go1WalkRoughRslrlPpo(Go1WalkFlatRslrlPpo):
        """Go1 robot walk on rough terrain - RSLRL PPO configuration.

        Uses larger network than flat terrain for more complex terrain handling.
        """

        def __post_init__(self):
            """Override network architecture for rough terrain."""
            super().__post_init__()

            # Override experiment name
            self.runner.experiment_name = "go1_rough_terrain_walk"

            # Override network architecture (larger for rough terrain)
            self.runner.actor.hidden_dims = [512, 256, 128]
            self.runner.critic.hidden_dims = [512, 256, 128]

    @rlcfg("go1-stairs-terrain-walk")
    @dataclass
    class Go1WalkStairsRslrlPpo(Go1WalkRoughRslrlPpo):
        """Go1 robot walk on stairs terrain - RSLRL PPO configuration.

        Uses same configuration as rough terrain since stairs also require complex handling.
        """

        def __post_init__(self):
            super().__post_init__()
            self.runner.experiment_name = "go1_stairs_terrain_walk"
