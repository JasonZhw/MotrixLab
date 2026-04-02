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

import abc
import dataclasses
from dataclasses import dataclass

import motrixsim as mtx
import numpy as np

from motrix_envs.base import ABEnv, EnvCfg


@dataclass
class NpEnvState:
    data: mtx.SceneData
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict

    @property
    def done(self) -> np.ndarray:
        """
        Check if the environment is done.
        """
        return np.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "NpEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self):
        num_envs = self.data.shape[0]
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape


class NpEnv(ABEnv):
    _model: mtx.SceneModel
    _cfg: EnvCfg
    _state: NpEnvState = None
    _render_spacing: float

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self._model = mtx.load_model(cfg.model_file)
        self._model.options.timestep = cfg.sim_dt
        self._render_spacing = cfg.render_spacing

    @property
    def model(self) -> mtx.SceneModel:
        """
        Get the scene model
        """
        return self._model

    @property
    def state(self) -> NpEnvState:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfg:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def render_spacing(self) -> bool:
        """
        Get the render spacing, with which the multi-envs will be rendered seperately in grid
        """
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> NpEnvState:
        """
        Create a new environment state
        """
        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        data = mtx.SceneData(self._model, batch=[self._num_envs])
        self._state = NpEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self):
        """
        Reset the environments that are done
        """
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not np.any(done):
            return

        np.putmask(state.info["steps"], done, 0)
        data = state.data[done]
        obs, info1 = self.reset(data)
        state.obs[done] = obs
        if info1:

            def replace_dict_values(dst, new_values, mask):
                for key, value in new_values.items():
                    if key not in dst:
                        dst[key] = value
                    else:
                        if isinstance(value, np.ndarray):
                            dst[key][mask] = value
                        elif isinstance(value, dict):
                            assert isinstance(dst[key], dict)
                            replace_dict_values(dst[key], value, mask)

            replace_dict_values(state.info, info1, done)

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        if not self._cfg.max_episode_steps:
            return
        self._state.truncated = self._state.info["steps"] >= self._cfg.max_episode_steps

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """
        Apply the action to the environment

        Args:
            actions (np.ndarray): The actions to apply
            state (NpEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        Update the environment state after physics step

        Args:
            state (NpEnvState): The environment state to update
        """

    @abc.abstractmethod
    def reset(
        self,
        data: mtx.SceneData,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for the done envs

        Args:
            data (mtx.SceneData): The scene data to reset

        Returns:
            tuple[np.ndarray, dict]: The initial observations and info after reset
        """
        pass

    def physics_step(self):
        # motrixsim.SceneModel.step only supports single step, so we loop
        for _ in range(self._cfg.sim_substeps):
            self._model.step(self._state.data)

    def _prev_physics_step(self):
        state = self._state
        state.reward.fill(0.0)
        state.terminated.fill(False)
        state.truncated.fill(False)

    def step(self, actions: np.ndarray) -> NpEnvState:
        if self._state is None:
            self.init_state()

        self._prev_physics_step()
        self._state = self.apply_action(actions, self._state)
        assert self._state is not None, "apply_action must return a valid NpEnvState"
        self.physics_step()
        self._state = self.update_state(self._state)
        self._state.info["steps"] += 1
        self._update_truncate()
        self._reset_done_envs()
        return self._state
