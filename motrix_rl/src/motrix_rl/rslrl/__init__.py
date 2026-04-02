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

"""RSLRL integration module for MotrixLab.

This module provides configuration classes and utilities for using RSLRL
(ETH Zurich's RL library) with MotrixLab.

The configuration structure matches rsl_rl's flat format with separate
actor and critic configs at the top level.
"""

from motrix_rl.rslrl.cfg import (
    RslRlActorCfg,
    RslRlCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslrlRunnerCfg,
)

__all__ = [
    "RslRlActorCfg",
    "RslRlCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslrlRunnerCfg",
    "field_override",
    "inherit_field",
    "configclass",
]
