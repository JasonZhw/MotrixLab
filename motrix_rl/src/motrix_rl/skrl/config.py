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


"""SKRL Configuration Classes

This module provides configuration classes for SKRL RL framework integration.
The configuration structure matches template/skrl_config.yaml with a hierarchical
design following the same pattern as RslrlConfig.

Configuration Hierarchy:
    SkrlCfg (top-level) -> SkrlRunnerCfg (runner-level)
        -> SkrlModelsCfg (models)
            -> SkrlPolicyCfg (policy model)
            -> SkrlValueCfg (value model)
        -> SkrlMemoryCfg (memory)
        -> SkrlAgentCfg (PPO agent)
            -> SkrlAgentExperimentCfg (experiment settings)
        -> SkrlTrainerCfg (trainer)
"""

import dataclasses
from dataclasses import dataclass, field


@dataclass
class SkrlPolicyCfg:
    """Configuration for SKRL policy (GaussianMixin) model.

    Corresponds to the policy section in template/skrl_config.yaml.
    """

    class_name: str = "GaussianMixin"
    clip_actions: bool = False
    clip_log_std: bool = True
    initial_log_std: float = 1.0
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    reduction: str = "sum"
    input: str = "STATES"
    hiddens: list[int] = field(default_factory=lambda: [256, 128, 64])
    hidden_activation: list[str] = field(default_factory=lambda: ["elu"])
    output: str = "ACTIONS"
    output_activation: str = ""
    output_scale: float = 1.0

    def _normalize_activations(self, num_layers: int) -> str | list[str]:
        """Normalize hidden_activation to match num_layers.

        SKRL requires either a single activation string (applied to all layers)
        or a list with length matching the number of layers.

        Args:
            num_layers: Number of hidden layers (len(self.hiddens))

        Returns:
            str or list[str] suitable for SKRL's network format

        Raises:
            ValueError: If activation list length > 1 and doesn't match num_layers
        """
        activations = self.hidden_activation

        # Empty list -> no activations
        if isinstance(activations, list) and len(activations) == 0:
            return [""] * num_layers

        # Single element list -> convert to string (SKRL will replicate)
        if isinstance(activations, list) and len(activations) == 1:
            return activations[0]

        # String -> return as-is (SKRL will replicate)
        if isinstance(activations, str):
            return activations

        # List with matching length -> use as-is
        if isinstance(activations, list) and len(activations) == num_layers:
            return activations

        # List with mismatched length > 1 -> raise error
        if isinstance(activations, list) and len(activations) > 1:
            raise ValueError(
                f"hidden_activation length ({len(activations)}) must match "
                f"the number of hidden layers ({num_layers}), or be a single value "
                f"to apply to all layers. Got hiddens={self.hiddens}, "
                f"hidden_activation={activations}"
            )

        return activations

    def to_network(self) -> tuple[list[dict], str]:
        """Convert configuration to SKRL's network and output format.

        Returns:
            (network, output) tuple where:
            - network: SKRL network definition list of dicts
            - output: SKRL output expression string (e.g., "tanh(ACTIONS)", "ONE")

        Examples:
            Policy with hiddens=[256,128,64], output_activation="tanh", output_scale=1.0:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "tanh(ACTIONS)"

            Value with hiddens=[256,128,64], output_activation="", output_scale=0.5:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "0.5 * ONE"
        """
        # Normalize activations to match hiddens length
        num_layers = len(self.hiddens)
        activations = self._normalize_activations(num_layers)

        # Build network definition
        network = [
            {
                "name": "net",
                "input": "STATES",
                "layers": self.hiddens,
                "activations": activations,
            }
        ]

        # Build output expression
        # Use output field directly (already in correct format)

        # Apply scale if not 1.0
        scale_prefix = f"{self.output_scale} * " if self.output_scale != 1.0 else ""

        # Apply activation if specified
        if self.output_activation:
            output = f"{scale_prefix}{self.output_activation}({self.output})"
        else:
            output = f"{scale_prefix}{self.output}"

        return network, output

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlValueCfg:
    """Configuration for SKRL value (DeterministicMixin) model.

    Corresponds to the value section in template/skrl_config.yaml.
    """

    class_name: str = "DeterministicMixin"
    clip_actions: bool = False
    input: str = "STATES"
    hiddens: list[int] = field(default_factory=lambda: [256, 128, 64])
    hidden_activation: list[str] = field(default_factory=lambda: ["elu"])
    output: str = "ONE"
    output_activation: str = ""
    output_scale: float = 1.0

    def _normalize_activations(self, num_layers: int) -> str | list[str]:
        """Normalize hidden_activation to match num_layers.

        SKRL requires either a single activation string (applied to all layers)
        or a list with length matching the number of layers.

        Args:
            num_layers: Number of hidden layers (len(self.hiddens))

        Returns:
            str or list[str] suitable for SKRL's network format

        Raises:
            ValueError: If activation list length > 1 and doesn't match num_layers
        """
        activations = self.hidden_activation

        # Empty list -> no activations
        if isinstance(activations, list) and len(activations) == 0:
            return [""] * num_layers

        # Single element list -> convert to string (SKRL will replicate)
        if isinstance(activations, list) and len(activations) == 1:
            return activations[0]

        # String -> return as-is (SKRL will replicate)
        if isinstance(activations, str):
            return activations

        # List with matching length -> use as-is
        if isinstance(activations, list) and len(activations) == num_layers:
            return activations

        # List with mismatched length > 1 -> raise error
        if isinstance(activations, list) and len(activations) > 1:
            raise ValueError(
                f"hidden_activation length ({len(activations)}) must match "
                f"the number of hidden layers ({num_layers}), or be a single value "
                f"to apply to all layers. Got hiddens={self.hiddens}, "
                f"hidden_activation={activations}"
            )

        return activations

    def to_network(self) -> tuple[list[dict], str]:
        """Convert configuration to SKRL's network and output format.

        Returns:
            (network, output) tuple where:
            - network: SKRL network definition list of dicts
            - output: SKRL output expression string (e.g., "tanh(ACTIONS)", "ONE")

        Examples:
            Policy with hiddens=[256,128,64], output_activation="tanh", output_scale=1.0:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "tanh(ACTIONS)"

            Value with hiddens=[256,128,64], output_activation="", output_scale=0.5:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "0.5 * ONE"
        """
        # Normalize activations to match hiddens length
        num_layers = len(self.hiddens)
        activations = self._normalize_activations(num_layers)

        # Build network definition
        network = [
            {
                "name": "net",
                "input": "STATES",
                "layers": self.hiddens,
                "activations": activations,
            }
        ]

        # Build output expression
        # Use output field directly (already in correct format)

        # Apply scale if not 1.0
        scale_prefix = f"{self.output_scale} * " if self.output_scale != 1.0 else ""

        # Apply activation if specified
        if self.output_activation:
            output = f"{scale_prefix}{self.output_activation}({self.output})"
        else:
            output = f"{scale_prefix}{self.output}"

        return network, output

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlModelsCfg:
    """Configuration for SKRL models section.

    Corresponds to the models section in template/skrl_config.yaml.
    """

    separate: bool = False
    policy: SkrlPolicyCfg = field(default_factory=SkrlPolicyCfg)
    value: SkrlValueCfg = field(default_factory=SkrlValueCfg)

    def to_dict(self) -> dict:
        """Convert to dict with nested configs."""
        return {
            "separate": self.separate,
            "policy": self.policy.to_dict(),
            "value": self.value.to_dict(),
        }


@dataclass
class SkrlMemoryCfg:
    """Configuration for SKRL memory.

    Corresponds to the memory section in template/skrl_config.yaml.
    """

    class_name: str = "RandomMemory"
    memory_size: int = -1  # -1: automatically determined

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlAgentExperimentCfg:
    """Experiment settings within agent config.

    Corresponds to the experiment subsection in template/skrl_config.yaml.
    """

    directory: str = "runs"
    experiment_name: str = ""
    write_interval: int = -1
    checkpoint_interval: int = -1


@dataclass
class SkrlAgentCfg:
    """Configuration for SKRL PPO agent.

    Corresponds to the agent section in template/skrl_config.yaml.
    Field names match PPO_DEFAULT_CONFIG from SKRL.
    """

    class_name: str = "PPO"
    rollouts: int = 32
    learning_epochs: int = 2
    mini_batches: int = 32
    discount_factor: float = 0.99
    lam: float = 0.95
    learning_rate: float = 1e-3
    learning_rate_scheduler: str = "KLAdaptiveLR"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {"kl_threshold": 0.008})
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 1.0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    clip_predicted_values: bool = True
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 2.0
    kl_threshold: int = 0
    rewards_shaper_scale: float = 1.0
    time_limit_bootstrap: bool = True
    experiment: SkrlAgentExperimentCfg = field(default_factory=SkrlAgentExperimentCfg)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for SKRL PPO agent.

        Returns:
            Dictionary representation matching SKRL's PPO agent configuration format.
            Maps class_name -> class and lam -> lambda for SKRL compatibility.

        Note:
            - Maps 'class_name' to 'class' (SKRL convention)
            - Maps 'lam' to 'lambda' (Python keyword conflict)
            - Converts nested experiment config to dict
            - Excludes state/value preprocessor fields (added dynamically during training)
        """
        # Build base configuration dict
        result = {
            "class": self.class_name,
            "rollouts": self.rollouts,
            "learning_epochs": self.learning_epochs,
            "mini_batches": self.mini_batches,
            "discount_factor": self.discount_factor,
            "lambda": self.lam,
            "learning_rate": self.learning_rate,
            "learning_rate_scheduler": self.learning_rate_scheduler,
            "learning_rate_scheduler_kwargs": self.learning_rate_scheduler_kwargs,
            "random_timesteps": self.random_timesteps,
            "learning_starts": self.learning_starts,
            "grad_norm_clip": self.grad_norm_clip,
            "ratio_clip": self.ratio_clip,
            "value_clip": self.value_clip,
            "clip_predicted_values": self.clip_predicted_values,
            "entropy_loss_scale": self.entropy_loss_scale,
            "value_loss_scale": self.value_loss_scale,
            "kl_threshold": self.kl_threshold,
            "rewards_shaper_scale": self.rewards_shaper_scale,
            "time_limit_bootstrap": self.time_limit_bootstrap,
            "experiment": {
                "directory": self.experiment.directory,
                "experiment_name": self.experiment.experiment_name,
                "write_interval": self.experiment.write_interval,
                "checkpoint_interval": self.experiment.checkpoint_interval,
            },
        }

        return result


@dataclass
class SkrlTrainerCfg:
    """Configuration for SKRL sequential trainer.

    Corresponds to the trainer section in template/skrl_config.yaml.
    """

    class_name: str = "SequentialTrainer"
    timesteps: int = 10000
    """
    The max number of batch env steps to run
    """

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlRunnerCfg:
    """Main SKRL runner configuration.

    This mirrors the structure in template/skrl_config.yaml.
    Follows the same pattern as RslrlRunnerCfg with nested configs
    and a to_dict() method for dictionary conversion.
    """

    seed: int = 42
    models: SkrlModelsCfg = field(default_factory=SkrlModelsCfg)
    memory: SkrlMemoryCfg = field(default_factory=SkrlMemoryCfg)
    agent: SkrlAgentCfg = field(default_factory=SkrlAgentCfg)
    trainer: SkrlTrainerCfg = field(default_factory=SkrlTrainerCfg)

    def to_dict(self) -> dict:
        """Convert config to dictionary for SKRL.

        Returns:
            Dictionary representation matching SKRL's expected format.
            Maps class_name -> class for all nested configs.

        Note:
            This method ensures that the output dictionary matches the exact
            structure of template/skrl_config.yaml, including the 'class' field
            names (instead of 'class_name' used in Python to avoid keyword conflicts).
        """
        result = {
            "seed": self.seed,
            "models": self.models.to_dict(),
            "memory": self.memory.to_dict(),
            "agent": self.agent.to_dict(),
            "trainer": self.trainer.to_dict(),
        }
        return result


@dataclass
class SkrlCfg:
    """Top-level SKRL configuration.

    Follows the same pattern as RslrlCfg with environment-level settings
    at the top level and runner configuration nested.
    """

    # Basic training parameters
    num_envs: int = 2048
    play_num_envs: int = 16

    runner: SkrlRunnerCfg = field(default_factory=SkrlRunnerCfg)

    def replace(self, **updates) -> "SkrlCfg":
        """Replace specified fields and return a new instance."""
        return dataclasses.replace(self, **updates)
