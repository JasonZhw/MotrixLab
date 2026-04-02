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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypeVar

from motrix_envs import registry as env_registry

logger = logging.getLogger(__name__)

TRLCfg = TypeVar("TRLCfg")


@dataclass
class EnvRlCfgs:
    cfgs: dict[str, dict[str, Type]] = field(default_factory=dict)
    """
    The RL configuration classes available for this environment.
    Structure: {rl_framework: {backend: config_class}}
    Example: {"skrl": {"jax": JaxConfig, "torch": TorchConfig}}
    """


# RL configuration registry. map from env name to EnvMeta
_rlcfgs: dict[str, EnvRlCfgs] = {}


def _register_rlcfg(env_name: str, rllib: str, backend: str, train_cfg_cls: Type):
    """
    Register a training configuration class for an environment, reinforcement learning framework, and backend.

    Args:
        env_name: Environment name
        rllib: RL framework name (e.g., "skrl")
        backend: Backend name (e.g., "jax", "torch")
        train_cfg_cls: Configuration class
    """
    if not env_registry.contains(env_name):
        raise ValueError(f"Environment '{env_name}' is not registered in env_registry.")

    logger.info(f"Registering RL config for env '{env_name}', RL framework '{rllib}', and backend '{backend}'")
    if env_name not in _rlcfgs:
        _rlcfgs[env_name] = EnvRlCfgs()
    if rllib not in _rlcfgs[env_name].cfgs:
        _rlcfgs[env_name].cfgs[rllib] = {}
    _rlcfgs[env_name].cfgs[rllib][backend] = train_cfg_cls


def _infer_framework_from_class(cls: Type) -> str:
    """Infer RL framework name from the class's parent class.

    Args:
        cls: Configuration class to inspect

    Returns:
        Framework name ("skrl" or "rslrl")

    Raises:
        ValueError: If framework cannot be determined from parent class
    """
    # Import here to avoid circular imports
    from motrix_rl.rslrl.cfg import RslrlCfg
    from motrix_rl.skrl.config import SkrlCfg

    # Check entire MRO (Method Resolution Order) for framework base classes
    for base in cls.__mro__:
        if base is SkrlCfg:
            return "skrl"
        elif base is RslrlCfg:
            return "rslrl"

    raise ValueError(
        f"Cannot infer RL framework from {cls.__name__}. Class must inherit from either SkrlCfg or RslrlCfg."
    )


def rlcfg(env_name: str, backend: str = None) -> Callable[[Type[TRLCfg]], Type[TRLCfg]]:
    """
    Decorator to register a training configuration class for an environment and backend.

    The RL framework (skrl/rslrl) is automatically inferred from the parent class.

    Args:
        env_name: Environment name
        backend: Backend name (e.g., "jax", "torch"). If None, registers for all backends.
    """

    def decorator(cls: Type[TRLCfg]) -> Type[TRLCfg]:
        # Infer framework from parent class
        rl_framework = _infer_framework_from_class(cls)

        backends = ["jax", "torch"] if backend is None else [backend]
        for b in backends:
            _register_rlcfg(env_name, rl_framework, b, cls)
        return cls

    return decorator


def default_rl_cfg(env_name: str, rllib: str, backend: str) -> Any:
    """
    Get the default training configuration for an environment, reinforcement learning framework, and backend.

    Args:
        env_name: Environment name
        rllib: RL framework name (e.g., "skrl")
        backend: Backend name (e.g., "jax", "torch")

    Returns:
        The configuration class instance. Will use backend-specific config if available,
        otherwise falls back to universal config (backend=None).
    """
    if env_name not in _rlcfgs:
        raise ValueError(f"Environment '{env_name}' is not registered.")
    meta: EnvRlCfgs = _rlcfgs.get(env_name)
    if rllib not in meta.cfgs:
        raise ValueError(f"RL framework '{rllib}' is not supported for environment '{env_name}'.")

    framework_configs = meta.cfgs[rllib]

    # Try to get backend-specific config first
    if backend in framework_configs:
        return framework_configs[backend]()

    # Fall back to universal config (backend=None) if backend-specific one is not found
    if None in framework_configs:
        return framework_configs[None]()

    # If no universal config exists, raise an error
    raise ValueError(
        f"No configuration found for environment '{env_name}', RL framework '{rllib}', backend '{backend}', \
            and no universal configuration available."
    )
