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

"""Tests for motrix_rl.utils.cfg_override function."""

import dataclasses
from dataclasses import dataclass

import pytest

from motrix_rl.utils import cfg_override


@dataclass
class NestedConfig:
    value: int = 10
    name: str = "default"


@dataclass
class MiddleConfig:
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    flag: bool = True


@dataclass
class RootConfig:
    middle: MiddleConfig = dataclasses.field(default_factory=MiddleConfig)
    count: int = 5
    label: str = "root"


class TestCfgOverride:
    """Tests for cfg_override function."""

    def test_nested_overrides(self):
        """Test overriding fields at all nesting levels."""
        cfg = RootConfig()

        # Top-level
        result = cfg_override(cfg, {"count": 42})
        assert result.count == 42
        assert result.label == "root"

        # One-level nested
        result = cfg_override(cfg, {"middle.flag": False})
        assert result.middle.flag is False

        # Deep nested
        result = cfg_override(cfg, {"middle.nested.value": 99})
        assert result.middle.nested.value == 99

    def test_multiple_overrides(self):
        """Test overriding multiple fields at different levels."""
        cfg = RootConfig()
        overrides = {
            "count": 42,
            "label": "modified",
            "middle.flag": False,
            "middle.nested.name": "custom",
        }
        result = cfg_override(cfg, overrides)

        assert result.count == 42
        assert result.label == "modified"
        assert result.middle.flag is False
        assert result.middle.nested.name == "custom"

    def test_type_and_list_fields(self):
        """Test overriding different field types."""
        cfg = RootConfig()

        # String
        result = cfg_override(cfg, {"label": "new_label"})
        assert result.label == "new_label"

        # Bool
        result = cfg_override(cfg, {"middle.flag": False})
        assert result.middle.flag is False

        # Float (Python dataclasses don't enforce types)
        result = cfg_override(cfg, {"count": 3.14})
        assert result.count == 3.14

        # List
        result = cfg_override(cfg, {"count": [1, 2, 3]})
        assert result.count == [1, 2, 3]

    def test_immutability_and_empty_overrides(self):
        """Test immutability and empty overrides."""
        cfg = RootConfig()
        original_count = cfg.count

        # Empty overrides returns original
        result = cfg_override(cfg, {})
        assert result is cfg

        # Original unchanged
        result = cfg_override(cfg, {"count": 42})
        assert cfg.count == original_count
        assert result.count == 42

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        cfg = RootConfig()

        # Non-existent top-level field
        with pytest.raises(KeyError, match="Invalid field"):
            cfg_override(cfg, {"nonexistent": 1})

        # Non-existent nested field
        with pytest.raises(KeyError, match="Invalid field"):
            cfg_override(cfg, {"middle.nonexistent": 1})

        # Non-existent intermediate path
        with pytest.raises(KeyError, match="Invalid path component"):
            cfg_override(cfg, {"invalid.path": 1})

        # Navigate into non-dataclass field
        with pytest.raises(TypeError, match="Cannot navigate into non-dataclass"):
            cfg_override(cfg, {"count.something": 1})

        # Non-dataclass cfg
        with pytest.raises(TypeError, match="cfg must be a dataclass"):
            cfg_override({"foo": "bar"}, {"foo": "baz"})

    def test_rslrl_config(self):
        """Test with actual RSLRL configuration classes."""
        from motrix_rl.rslrl.cfg import RslrlCfg

        cfg = RslrlCfg()
        overrides = {
            "num_envs": 4096,
            "runner.seed": 123,
            "runner.algorithm.num_learning_epochs": 10,
            "runner.algorithm.learning_rate": 1e-4,
            "runner.actor.hidden_dims": [512, 256, 128],
        }
        result = cfg_override(cfg, overrides)

        assert result.num_envs == 4096
        assert result.play_num_envs == 16  # Unchanged
        assert result.runner.seed == 123
        assert result.runner.algorithm.num_learning_epochs == 10
        assert result.runner.algorithm.learning_rate == 1e-4
        assert result.runner.actor.hidden_dims == [512, 256, 128]
