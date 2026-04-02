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

from motrix_envs import registry
from motrix_envs.basic.reacher.cfg import ReacherEnvCfg
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("dm-reacher", "np")
class Reacher2DEnv(NpEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: ReacherEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)

        self._target_size = cfg.target_size
        self._finger = self._model.get_link("finger")
        self._joint_limits = self._model.joint_limits

        self._target_body = self._model.get_body("target")
        self._target_xyz = np.zeros((num_envs, 3), dtype=np.float32)
        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        num_obs = self._model.num_dof_pos + 2 + self._model.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        low, high = self._model.actuator_ctrl_limits
        self._action_space = gym.spaces.Box(low, high, (self._model.num_actuators,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions, state):
        state.data.actuator_ctrls = actions
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = data.dof_pos
        qvel = data.dof_vel
        finger_xy = self._finger.get_pose(data)[:, :2]
        to_target = self._target_xyz[:, :2] - finger_xy
        return np.concatenate([qpos, to_target, qvel], axis=-1)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)
        finger_xy = self._finger.get_pose(data)[:, :2]
        dist = np.linalg.norm(self._target_xyz[:, :2] - finger_xy, axis=-1)
        rwd = reward.tolerance(
            dist, bounds=(0.0, self._target_size), margin=self._target_size, value_at_margin=0.0, sigmoid="linear"
        )
        terminated = np.isnan(obs).any(axis=-1)
        rwd[terminated] = 0.0

        state.info["Reward"] = {"distance": dist, "tolerance": rwd.copy()}

        return state.replace(obs=obs, reward=rwd, terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """Reset environment with randomized target position in xy plane (z=0)."""
        data.reset(self._model)
        num_reset = data.shape[0]

        dof_pos = np.zeros((num_reset, self._model.num_dof_pos))
        dof_pos[:, 0] = np.random.uniform(-np.pi, np.pi, size=(num_reset,))
        dof_pos[:, 1] = np.random.uniform(-np.pi, np.pi, size=(num_reset,))
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        target_x = np.random.uniform(-0.15, 0.15, size=(num_reset,))
        target_y = np.random.uniform(0.15, 0.15, size=(num_reset,))

        target_dof_pos = np.stack([target_x, target_y], axis=-1)

        self._target_body.set_dof_pos(data, target_dof_pos)

        self._model.forward_kinematic(data)

        target_pose = self._target_body.get_pose(data)
        self._target_xyz = target_pose.copy()
        self._target_xyz[:, 2] = 0.0

        obs = self._get_obs(data)
        rewards = {"distance": np.zeros((num_reset,)), "tolerance": np.zeros((num_reset,))}
        info = {
            "Reward": rewards,
        }

        return obs, info
