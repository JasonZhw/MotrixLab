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

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class DeviceSupports:
    torch: bool = False
    torch_gpu: bool = False
    jax: bool = False
    jax_gpu: bool = False


def _check_gpu_available_for_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros((1,)).cuda().numpy(force=True)
        return True
    except Exception:
        return False


def get_device_supports() -> DeviceSupports:
    supports = DeviceSupports()
    try:
        import torch  # noqa: F401

        supports.torch = True
        supports.torch_gpu = _check_gpu_available_for_torch()
    except ImportError:
        pass

    try:
        import jax  # noqa: F401

        supports.jax = True
        from jax.lib import xla_bridge

        platform = xla_bridge.get_backend().platform
        if platform == "gpu":
            supports.jax_gpu = True
    except ImportError:
        pass

    return supports


def class_to_dict(obj) -> dict | list | Any:
    """Recursively convert a dataclass to a dictionary.

    Args:
        obj: The object to convert (dataclass, list, dict, or primitive)

    Returns:
        Dictionary representation with nested dataclasses recursively converted
    """
    if dataclasses.is_dataclass(obj):
        return {k: class_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def cfg_override(cfg: T, overrides: dict[str, Any]) -> T:
    """Override dataclass fields using dot-notation path keys.

    This function creates a new dataclass instance with specified field values
    overridden, leaving the original config unchanged. Nested dataclasses are
    handled using dot notation in the key path.

    Args:
        cfg: The original dataclass configuration object
        overrides: Dictionary with path keys (e.g., "runner.seed", "num_envs")
                   where each key is a dot-separated path to the field to override

    Returns:
        A new dataclass instance with overrides applied

    Raises:
        KeyError: If a path key is invalid or references a non-existent field
        TypeError: If an intermediate field is not a dataclass or if a value
                   type doesn't match the expected field type

    Examples:
        >>> from motrix_rl.rslrl.cfg import RslrlCfg
        >>> base_cfg = RslrlCfg()
        >>> overrides = {
        ...     "num_envs": 4096,
        ...     "runner.seed": 123,
        ...     "runner.algorithm.num_learning_epochs": 10,
        ... }
        >>> new_cfg = cfg_override(base_cfg, overrides)
        >>> assert new_cfg.num_envs == 4096
        >>> assert new_cfg.runner.seed == 123
        >>> assert new_cfg.runner.algorithm.num_learning_epochs == 10
    """
    if not overrides:
        return cfg

    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"cfg must be a dataclass, got {type(cfg).__name__}")

    # Group overrides by their parent paths to apply them efficiently
    # Structure: {parent_path: {field_name: value}}
    # For "num_envs": parent_path=[], field_name="num_envs"
    # For "runner.seed": parent_path=["runner"], field_name="seed"
    override_tree: dict[tuple[str, ...], dict[str, Any]] = {}

    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 1:
            # Top-level field
            parent_path = tuple()
            field_name = parts[0]
        else:
            # Nested field
            parent_path = tuple(parts[:-1])
            field_name = parts[-1]

        if parent_path not in override_tree:
            override_tree[parent_path] = {}
        override_tree[parent_path][field_name] = value

    # Apply overrides from deepest to shallowest to minimize object copies
    sorted_paths = sorted(override_tree.keys(), key=lambda p: len(p), reverse=True)

    def apply_overrides_at_path(
        obj: Any, path: tuple[str, ...], field_overrides: dict[str, Any], parent_path: tuple[str, ...] = ()
    ) -> Any:
        """Apply overrides to an object at a specific path.

        Args:
            obj: The current object (dataclass or primitive)
            path: Tuple of field names to navigate through
            field_overrides: Dict of field names to values to apply at the target
            parent_path: The full path from the root (for error messages)

        Returns:
            New object with overrides applied
        """
        if not path:
            # We're at the target - apply the overrides
            if not dataclasses.is_dataclass(obj):
                # Build the full path for the error message
                full_path_parts = list(parent_path) if parent_path else ["(root)"]
                raise TypeError(
                    f"Cannot navigate into non-dataclass field '{full_path_parts[-1]}' of type {type(obj).__name__}"
                )

            # Validate field names exist - use set difference for efficiency
            obj_fields = {f.name for f in dataclasses.fields(obj)}
            invalid_fields = set(field_overrides.keys()) - obj_fields
            if invalid_fields:
                raise KeyError(
                    f"Invalid fields {sorted(invalid_fields)} for {type(obj).__name__}. "
                    f"Valid fields: {sorted(obj_fields)}"
                )

            # Use dataclasses.replace to create a new instance with overrides
            return dataclasses.replace(obj, **field_overrides)

        # Need to navigate deeper - recurse to modify nested dataclass
        if not dataclasses.is_dataclass(obj):
            raise TypeError(f"Cannot navigate into non-dataclass field '{path[0]}' of type {type(obj).__name__}")

        # Validate the navigation path exists
        obj_fields = {f.name: f for f in dataclasses.fields(obj)}
        if path[0] not in obj_fields:
            valid_fields = sorted(obj_fields.keys())
            raise KeyError(f"Invalid path component '{path[0]}' for {type(obj).__name__}. Valid fields: {valid_fields}")

        # Get the nested object
        nested_obj = getattr(obj, path[0])

        # Recursively apply overrides to the nested object
        new_nested_obj = apply_overrides_at_path(nested_obj, path[1:], field_overrides, parent_path + (path[0],))

        # Return a new instance of the current object with the nested field replaced
        return dataclasses.replace(obj, **{path[0]: new_nested_obj})

    # Start with the original cfg and apply each group of overrides
    result = cfg
    for path in sorted_paths:
        field_overrides = override_tree[path]
        result = apply_overrides_at_path(result, path, field_overrides, parent_path=path)

    return result
