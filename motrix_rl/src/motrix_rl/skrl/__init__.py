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

from datetime import datetime

LOG_DIR_PREFIX = "runs"


def get_log_dir(env_name: str, rllib: str = "skrl", agent_name: str = "PPO") -> str:
    """Get the log directory for the given environment name and RL framework.

    Args:
        env_name: Name of the environment
        rllib: RL framework name (e.g., "skrl", "rslrl")
        agent_name: Name of the agent (e.g., "PPO")

    Returns:
        Log directory path:
        - For SKRL: runs/{env_name}/{rllib}/ (SKRL creates its own timestamp subdirectory)
        - For RSLRL: runs/{env_name}/{rllib}/{time}_{agent}/ (RSLRL doesn't create subdirectories)
    """
    if rllib == "skrl":
        # SKRL creates its own timestamp subdirectory, so we just provide the base path
        return f"{LOG_DIR_PREFIX}/{env_name}/{rllib}"
    else:
        # RSLRL doesn't create subdirectories, so we add the timestamp here
        now = datetime.now()
        time_str = now.strftime("%y-%m-%d_%H-%M-%S")
        microseconds = now.microsecond
        time_str = f"{time_str}-_{microseconds:05d}"
        return f"{LOG_DIR_PREFIX}/{env_name}/{rllib}/{time_str}_{agent_name}"
