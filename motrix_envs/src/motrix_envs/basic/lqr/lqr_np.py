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

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim.render import Color

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import LqrBaseCfg


def _normalize_actions(actions: np.ndarray, num_envs: int, num_actuators: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        if num_envs != 1 or actions.shape[0] != num_actuators:
            raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}) or ({num_actuators},).")
        actions = actions.reshape(1, num_actuators)
    if actions.shape != (num_envs, num_actuators):
        raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}), got {actions.shape}.")
    return np.ascontiguousarray(actions)


@registry.env("dm-lqr-2-1", "np")
@registry.env("dm-lqr-6-2", "np")
class LqrEnv(NpEnv):
    _cfg: LqrBaseCfg

    def __init__(self, cfg: LqrBaseCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._nq = int(self._model.num_dof_pos)
        self._nv = int(self._model.num_dof_vel)
        self._nu = int(self._model.num_actuators)

        if self._nq != cfg.expected_nq or self._nv != cfg.expected_nq:
            raise ValueError(f"LQR model mismatch: expected nq=nv={cfg.expected_nq}, got nq={self._nq}, nv={self._nv}.")
        if self._nu != cfg.expected_nu:
            raise ValueError(f"LQR model mismatch: expected nu={cfg.expected_nu}, got nu={self._nu}.")

        obs_dim = self._nq + self._nv
        self._action_low = np.asarray(self._model.actuator_ctrl_limits[0], dtype=np.float32)
        self._action_high = np.asarray(self._model.actuator_ctrl_limits[1], dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._action_space = gym.spaces.Box(self._action_low, self._action_high, (self._nu,), dtype=np.float32)
        self._rope_geom_pairs = [
            (self._model.get_geom(f"geom_{i}"), self._model.get_geom(f"geom_{i + 1}")) for i in range(self._nq - 1)
        ]
        self._rope_color = Color.rgb(0.85, 0.75, 0.65)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        actions = _normalize_actions(actions, self._num_envs, self._nu)
        state.data.actuator_ctrls = np.clip(actions, self._action_low, self._action_high)
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = np.asarray(data.dof_pos, dtype=np.float32)
        qvel = np.asarray(data.dof_vel, dtype=np.float32)
        return np.concatenate([qpos, qvel], axis=-1)

    def draw_gizmos(self, gizmos, render_offsets: np.ndarray) -> None:
        if self.state is None:
            return

        offsets = np.asarray(render_offsets, dtype=np.float32)
        data = self.state.data
        gizmos.line_width = 6.0
        for geom_a, geom_b in self._rope_geom_pairs:
            start = np.asarray(geom_a.get_pose(data), dtype=np.float32)[..., :3]
            end = np.asarray(geom_b.get_pose(data), dtype=np.float32)[..., :3]
            for env_i in range(self._num_envs):
                gizmos.draw_line(
                    start[env_i] + offsets[env_i],
                    end[env_i] + offsets[env_i],
                    color=self._rope_color,
                )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        qpos = np.asarray(data.dof_pos, dtype=np.float32)
        qvel = np.asarray(data.dof_vel, dtype=np.float32)
        ctrl = np.asarray(data.actuator_ctrls, dtype=np.float32)

        obs = self._get_obs(data)
        out_of_bounds = np.any(np.abs(qpos) > self._cfg.boundary_position_limit, axis=-1)
        out_of_bounds |= np.any(np.abs(qvel) > self._cfg.boundary_velocity_limit, axis=-1)
        position_norm = np.linalg.norm(qpos, axis=-1)
        velocity_norm = np.linalg.norm(qvel, axis=-1)
        state_cost = 0.5 * np.sum(np.square(qpos), axis=-1)
        velocity_cost = 0.5 * self._cfg.velocity_cost_coef * np.sum(np.square(qvel), axis=-1)
        control_cost = 0.5 * self._cfg.control_cost_coef * np.sum(np.square(ctrl), axis=-1)
        success = (position_norm <= self._cfg.success_position_tol) & (velocity_norm <= self._cfg.success_velocity_tol)
        success &= ~out_of_bounds
        success_reward = self._cfg.success_bonus * success.astype(np.float32)
        boundary_penalty = self._cfg.out_of_bounds_penalty * out_of_bounds.astype(np.float32)
        reward = 1.0 - (state_cost + velocity_cost + control_cost) + success_reward - boundary_penalty

        terminated = success | out_of_bounds
        terminated |= np.isnan(obs).any(axis=-1)
        terminated |= np.isnan(ctrl).any(axis=-1)

        state.info["metrics"] = {
            "position_norm": position_norm.astype(np.float32),
            "velocity_norm": velocity_norm.astype(np.float32),
            "success": success.astype(np.float32),
            "out_of_bounds": out_of_bounds.astype(np.float32),
        }
        state.info["Reward"] = {
            "state_cost": (-state_cost).astype(np.float32),
            "velocity_cost": (-velocity_cost).astype(np.float32),
            "control_cost": (-control_cost).astype(np.float32),
            "success_bonus": success_reward.astype(np.float32),
            "out_of_bounds_penalty": (-boundary_penalty).astype(np.float32),
        }

        return state.replace(
            obs=obs,
            reward=reward.astype(np.float32),
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num_envs = int(data.shape[0])

        qpos = np.random.standard_normal((num_envs, self._nq)).astype(np.float32)
        norms = np.linalg.norm(qpos, axis=-1, keepdims=True)
        zero_norm = norms[:, 0] < 1e-8
        if np.any(zero_norm):
            qpos[zero_norm, 0] = 1.0
            norms = np.linalg.norm(qpos, axis=-1, keepdims=True)
        qpos *= self._cfg.reset_position_norm / np.clip(norms, 1e-8, None)

        qvel = np.zeros((num_envs, self._nv), dtype=np.float32)
        data.set_dof_pos(qpos, self._model)
        data.set_dof_vel(qvel)

        self._model.forward_kinematic(data)

        return np.concatenate([qpos, qvel], axis=-1), {}
