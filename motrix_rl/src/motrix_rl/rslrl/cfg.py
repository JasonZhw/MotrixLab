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


"""PPO Agent Configuration for RSLRL

This module provides configuration classes for PPO agents using the
RSLRL (ETH Zurich RL library) framework.

The configuration structure matches rsl_rl's flat format with separate
actor and critic configs at the top level.
"""

from dataclasses import dataclass, field
from typing import Literal

from motrix_rl.utils import class_to_dict


@dataclass
class RslRlActorCfg:
    """Configuration for the actor network."""

    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    obs_normalization: bool = True
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: Literal["scalar", "log"] = "scalar"
    state_dependent_std: bool = False


@dataclass
class RslRlCriticCfg:
    """Configuration for the critic network."""

    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    obs_normalization: bool = True
    stochastic: bool = False


@dataclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    optimizer: str = "adam"
    learning_rate: float = 3e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    value_loss_coef: float = 1.0
    clip_param: float = 0.2
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.008
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None


@dataclass
class RslrlRunnerCfg:
    """Configuration matching rsl_rl's flat structure.

    This configuration provides separate actor and critic configs at the top level,
    matching the structure expected by rsl_rl's OnPolicyRunner.
    """

    # Runner settings
    class_name: str = "OnPolicyRunner"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 50
    experiment_name: str = "experiment"
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"

    # Observation groups
    obs_groups: dict[str, list[str]] = field(default_factory=lambda: {"actor": ["policy"], "critic": ["policy"]})

    # Network configs - TOP LEVEL
    actor: RslRlActorCfg = field(default_factory=RslRlActorCfg)
    critic: RslRlCriticCfg = field(default_factory=RslRlCriticCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)

    def to_dict(self) -> dict:
        """Convert config to dictionary for OnPolicyRunner.

        Returns:
            Dictionary representation matching rsl_rl's expected format.
        """
        return class_to_dict(self)


@dataclass
class RslrlCfg:
    """Configuration for RSLRL."""

    num_envs: int = 2048
    play_num_envs: int = 16
    runner: RslrlRunnerCfg = field(default_factory=RslrlRunnerCfg)
