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
from __future__ import annotations

import numpy as np


def raw_action_to_close_ratio(raw_gripper_action: np.ndarray, use_sigmoid: bool) -> np.ndarray:
    raw = np.asarray(raw_gripper_action, dtype=np.float32)
    if bool(use_sigmoid):
        return 1.0 / (1.0 + np.exp(-raw))
    return np.clip((raw + 1.0) * 0.5, 0.0, 1.0)


def binary_hysteresis_step(
    *,
    close_ratio: np.ndarray,
    prev_closed: np.ndarray,
    steps_since_switch: np.ndarray,
    close_on_threshold: float,
    open_off_threshold: float,
    min_switch_interval_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    close_ratio = np.asarray(close_ratio, dtype=np.float32)
    prev_closed = np.asarray(prev_closed, dtype=bool)
    steps_since_switch = np.asarray(steps_since_switch, dtype=np.int32)

    close_on = float(np.clip(close_on_threshold, 0.0, 1.0))
    open_off = float(np.clip(open_off_threshold, 0.0, 1.0))
    if open_off > close_on:
        open_off = close_on
    min_steps = max(int(min_switch_interval_steps), 0)

    can_switch = steps_since_switch >= min_steps
    want_close = close_ratio > close_on
    want_open = close_ratio < open_off

    next_closed = prev_closed.copy()
    next_closed = np.where(
        np.logical_and(np.logical_not(prev_closed), np.logical_and(can_switch, want_close)),
        True,
        next_closed,
    )
    next_closed = np.where(
        np.logical_and(prev_closed, np.logical_and(can_switch, want_open)),
        False,
        next_closed,
    )
    switched = next_closed != prev_closed
    return next_closed.astype(bool), switched.astype(bool)
