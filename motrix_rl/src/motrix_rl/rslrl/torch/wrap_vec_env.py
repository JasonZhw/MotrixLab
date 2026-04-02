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

"""VecEnv wrapper for adapting NpEnv to RSLRL's VecEnv interface."""

import numpy as np
import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict

from motrix_envs.np.env import NpEnv


class RslrlNpEnvWrap(VecEnv):
    """Adapter class that wraps NpEnv to RSLRL's VecEnv interface.

    RSLRL expects a VecEnv interface with specific methods for stepping,
    resetting, and accessing observations. This adapter converts between
    NpEnv's NpEnvState format and RSLRL's expected format.
    """

    def __init__(self, env: NpEnv, device: torch.device):
        """Initialize the VecEnv adapter.

        Args:
            env: The NpEnv instance to wrap
            device: PyTorch device for tensors
        """
        self._env = env
        self._device = device
        self._state = None
        self._num_envs = env.num_envs
        self._viewer = None  # Will be initialized lazily when render() is called

        # Set max_episode_length from env config
        self.max_episode_length = self._env.cfg.max_episode_steps if self._env.cfg.max_episode_steps else 10000

        # Episode length buffer for tracking
        self.episode_length_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Configuration dict for RSLRL logger
        self.cfg = {
            "env_name": self._env.cfg.__class__.__name__,
        }

        # Initialize the environment state
        self.reset()

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def num_obs(self) -> int:
        """Size of observation space."""
        return self._env.observation_space.shape[0]

    @property
    def num_actions(self) -> int:
        """Size of action space."""
        return self._env.action_space.shape[0]

    @property
    def device(self) -> torch.device:
        """PyTorch device for tensors."""
        return self._device

    @property
    def unwrapped(self) -> "RslrlNpEnvWrap":
        """Return the unwrapped environment (self for this wrapper)."""
        return self

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # Convert torch actions to numpy
        actions_np = actions.cpu().numpy()

        # Step the environment
        state = self._env.step(actions_np)
        self._state = state

        # Update episode length buffer
        self.episode_length_buf += 1
        # Reset episode length for done environments
        dones_np = state.done.astype(bool)
        self.episode_length_buf[dones_np] = 0

        # Convert to torch tensors
        obs_tensor = torch.from_numpy(state.obs).to(self._device)
        rewards = torch.from_numpy(state.reward).to(self._device)

        # Merge terminated and truncated into dones
        dones = torch.from_numpy(state.done.astype(np.float32)).to(self._device)

        # Create TensorDict for observations
        obs = TensorDict({"policy": obs_tensor}, batch_size=[self._num_envs], device=self._device)

        # Build extras dict (RSLRL calls it "extras" not "infos")
        extras = {}
        if "time_outs" in state.info:
            extras["time_outs"] = torch.from_numpy(state.info["time_outs"]).to(self._device)

        return obs, rewards, dones, extras

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments.

        Returns:
            Tuple of (observations, extras)
            - observations: TensorDict with observation groups
            - extras: dict with episode information
        """
        state = self._env.init_state()
        self._state = state

        # Reset episode length buffer
        self.episode_length_buf.zero_()

        obs_tensor = torch.from_numpy(state.obs).to(self._device)

        # Create TensorDict for observations
        obs = TensorDict({"policy": obs_tensor}, batch_size=[self._num_envs], device=self._device)

        # Build extras dict
        extras = {}

        return obs, extras

    def get_observations(self) -> TensorDict:
        """Get current observations without stepping the environment.

        Returns:
            Current observations as TensorDict
        """
        if self._state is None:
            obs, _ = self.reset()
            return obs

        obs_tensor = torch.from_numpy(self._state.obs).to(self._device)
        obs = TensorDict({"policy": obs_tensor}, batch_size=[self._num_envs], device=self._device)
        return obs

    def render(self) -> None:
        """Render the environment.

        For NpEnv, this triggers the motrixsim viewer to display the scene.
        The viewer window must be kept alive by calling this method regularly.
        """
        # Initialize viewer on first call
        if self._viewer is None:
            from motrix_envs.np.renderer import NpRenderer

            self._viewer = NpRenderer(env=self._env)

        # Render the current state
        self._viewer.render()
