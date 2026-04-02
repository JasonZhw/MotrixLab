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
from motrix_envs.basic.finger.cfg import FingerBaseCfg
from motrix_envs.np.env import NpEnv, NpEnvState


def _sanitize_joint_limits(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.where(np.isfinite(low), low, -np.pi)
    high = np.where(np.isfinite(high), high, np.pi)
    return low, high


class FingerEnv(NpEnv):
    _cfg: FingerBaseCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: FingerBaseCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._cfg = cfg

        self._spinner = self._model.get_link("spinner")
        self._tip_site = self._model.get_site("tip")
        self._target_site = self._model.get_site("target")
        self._cap1 = self._model.get_geom("cap1")
        self._touchtop_site = self._model.get_site("touchtop")
        self._touchbottom_site = self._model.get_site("touchbottom")

        self._joint_limit_low, self._joint_limit_high = _sanitize_joint_limits(*self._model.joint_limits)

        # Cache joint dof indices
        self._prox_qpos_i = self._joint_pos_index("proximal")
        self._dist_qpos_i = self._joint_pos_index("distal")
        self._hinge_qpos_i = self._joint_pos_index("hinge")
        self._hinge_qvel_i = self._joint_vel_index("hinge")
        self._prox_qvel_i = self._joint_vel_index("proximal")
        self._dist_qvel_i = self._joint_vel_index("distal")

        self._target_xyz = np.zeros((num_envs, 3), dtype=np.float32)
        self._target_radius = float(cfg.target_radius)
        self._spin_vel_threshold = float(cfg.spin_velocity_threshold)

        self._init_obs_space()
        self._init_action_space()

    def _joint_pos_index(self, joint_name: str) -> int:
        joint_index = self._model.get_joint_index(joint_name)
        return int(self._model.joint_dof_pos_indices[joint_index])

    def _joint_vel_index(self, joint_name: str) -> int:
        joint_index = self._model.get_joint_index(joint_name)
        return int(self._model.joint_dof_vel_indices[joint_index])

    def _init_obs_space(self):
        raise NotImplementedError

    def _init_action_space(self):
        low, high = self._model.actuator_ctrl_limits
        self._action_space = gym.spaces.Box(low, high, (self._model.num_actuators,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        # Keep track of actions for reward shaping (e.g., smoothness penalties)
        if "actions" not in state.info:
            state.info["actions"] = np.zeros_like(actions, dtype=np.float32)
        if "last_actions" not in state.info:
            state.info["last_actions"] = np.zeros_like(actions, dtype=np.float32)
        state.info["last_actions"] = state.info["actions"]
        state.info["actions"] = actions
        state.data.actuator_ctrls = actions
        return state

    def _touch(self, data: mtx.SceneData) -> np.ndarray:
        top = np.asarray(self._model.get_sensor_value("touchtop", data)).reshape(data.shape[0], -1)[:, 0]
        bottom = np.asarray(self._model.get_sensor_value("touchbottom", data)).reshape(data.shape[0], -1)[:, 0]
        return np.log1p(np.stack([top, bottom], axis=-1))

    def _tip_position_xz(self, data: mtx.SceneData) -> np.ndarray:
        tip_xyz = self._tip_site.get_position(data)
        spinner_xyz = self._spinner.get_position(data)
        return (tip_xyz - spinner_xyz)[:, [0, 2]]

    def _target_position_xz(self, data: mtx.SceneData) -> np.ndarray:
        spinner_xyz = self._spinner.get_position(data)
        return (self._target_xyz - spinner_xyz)[:, [0, 2]]

    def _dist_to_target(self, data: mtx.SceneData) -> np.ndarray:
        # Signed distance to the target surface. Negative means inside.
        tip_xyz = self._tip_site.get_position(data)
        dist = np.linalg.norm((self._target_xyz - tip_xyz)[:, [0, 2]], axis=-1)
        return dist - self._target_radius

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        raise NotImplementedError

    def update_state(self, state: NpEnvState) -> NpEnvState:
        raise NotImplementedError

    def _maybe_init_target_freejoint(self, dof_pos: np.ndarray) -> slice | None:
        # Optional freejoint-backed target visualization body (7 qpos: xyz + quat).
        try:
            self._model.get_geom("target_geom")
            target_free_pos = slice(self._model.num_dof_pos - 7, self._model.num_dof_pos)
            dof_pos[:, target_free_pos] = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return target_free_pos
        except Exception:
            return None

    def _reset_collision_free_joint_angles(
        self, data: mtx.SceneData, dof_pos: np.ndarray, target_free_pos: slice | None
    ):
        # Randomize joint angles with a collision-free rejection sampler (dm_control-style).
        # The MotrixSim joint_limits are per-joint (not per-DOF), so we explicitly fill each DOF.
        num = int(data.shape[0])
        max_attempts = int(getattr(self._cfg, "reset_collision_free_attempts", 200))
        pending = np.ones((num,), dtype=bool)
        for _ in range(max_attempts):
            if not pending.any():
                break

            num_pending = int(pending.sum())
            for joint_name in ("proximal", "distal"):
                j = self._model.get_joint_index(joint_name)
                dof_i = self._joint_pos_index(joint_name)
                low = float(self._joint_limit_low[j])
                high = float(self._joint_limit_high[j])
                dof_pos[pending, dof_i] = np.random.uniform(low=low, high=high, size=(num_pending,)).astype(np.float32)

            # Sample hinge position (unlimited in model)
            dof_pos[pending, self._hinge_qpos_i] = np.random.uniform(
                low=-np.pi, high=np.pi, size=(num_pending,)
            ).astype(np.float32)

            if target_free_pos is not None:
                dof_pos[:, target_free_pos] = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            data.set_dof_pos(dof_pos, self._model)
            data.set_dof_vel(np.zeros((num, self._model.num_dof_vel), dtype=np.float32))
            self._model.forward_kinematic(data)
            pending = self._model.get_contact_query(data).num_contacts > 0

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


@registry.env("dm-finger-spin", "np")
class FingerSpinEnv(FingerEnv):
    def _init_obs_space(self):
        # Match dm_control's observation dict, but flatten into a vector.
        # Spin: position(4) + velocity(3) + touch(2) = 9
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = data.dof_pos
        qvel = data.dof_vel
        position = np.concatenate(
            [
                qpos[:, [self._prox_qpos_i, self._dist_qpos_i]],
                self._tip_position_xz(data),
            ],
            axis=-1,
        )
        velocity = qvel[:, [self._prox_qvel_i, self._dist_qvel_i, self._hinge_qvel_i]]
        touch = self._touch(data)
        return np.concatenate([position, velocity, touch], axis=-1).astype(np.float32)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)
        terminated = np.isnan(obs).any(axis=-1)

        hinge_velocity = data.dof_vel[:, self._hinge_qvel_i]
        spin_sparse = (hinge_velocity <= -self._spin_vel_threshold).astype(np.float32)

        if self._cfg.reward_mode == "shaped":
            # Dense reward to help PPO learn: encourage fast negative hinge velocity.
            # Range: [0, 1] roughly, with 1 around reaching the threshold.
            spin = np.clip((-hinge_velocity) / self._spin_vel_threshold, 0.0, 1.0).astype(np.float32)
            if self._cfg.shaped_reward_beta != 1.0:
                spin = np.power(spin, self._cfg.shaped_reward_beta, dtype=np.float32)
        else:
            spin = spin_sparse

        touch_raw = np.zeros((data.shape[0],), dtype=np.float32)
        touch_bonus = np.zeros((data.shape[0],), dtype=np.float32)
        approach_dist = np.zeros((data.shape[0],), dtype=np.float32)
        approach_reward = np.zeros((data.shape[0],), dtype=np.float32)

        if self._cfg.reward_mode == "shaped":
            if float(getattr(self._cfg, "spin_touch_bonus_scale", 0.0)) > 0.0:
                top = np.asarray(self._model.get_sensor_value("touchtop", data)).reshape(data.shape[0], -1)[:, 0]
                bottom = np.asarray(self._model.get_sensor_value("touchbottom", data)).reshape(data.shape[0], -1)[:, 0]
                touch_raw = (top + bottom).astype(np.float32)
                touch_bonus = (
                    float(self._cfg.spin_touch_bonus_scale)
                    * np.tanh(touch_raw / float(max(self._cfg.spin_touch_bonus_tanh_scale, 1e-6)))
                ).astype(np.float32)

            if float(getattr(self._cfg, "spin_approach_reward_scale", 0.0)) > 0.0:
                spinner_xyz = self._spinner.get_position(data)
                top_xyz = self._touchtop_site.get_position(data)
                bottom_xyz = self._touchbottom_site.get_position(data)
                top_dist = np.linalg.norm((top_xyz - spinner_xyz)[:, [0, 2]], axis=-1)
                bottom_dist = np.linalg.norm((bottom_xyz - spinner_xyz)[:, [0, 2]], axis=-1)
                approach_dist = np.minimum(top_dist, bottom_dist).astype(np.float32)
                sigma = float(max(self._cfg.spin_approach_sigma, 1e-6))
                approach_reward = (float(self._cfg.spin_approach_reward_scale) * np.exp(-approach_dist / sigma)).astype(
                    np.float32
                )

            spin = np.clip(spin + touch_bonus + approach_reward, 0.0, 1.0).astype(np.float32)

        rwd = spin
        state.info["Reward"] = {
            "hinge_velocity": hinge_velocity.copy(),
            "spin": spin.copy(),
            "spin_sparse": spin_sparse.copy(),
            "touch_raw": touch_raw.copy(),
            "touch_bonus": touch_bonus.copy(),
            "approach_dist": approach_dist.copy(),
            "approach_reward": approach_reward.copy(),
        }

        rwd[terminated] = 0.0
        return state.replace(obs=obs, reward=rwd, terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num = int(data.shape[0])
        dof_pos = np.zeros((num, self._model.num_dof_pos), dtype=np.float32)
        target_free_pos = self._maybe_init_target_freejoint(dof_pos)
        self._reset_collision_free_joint_angles(data, dof_pos, target_free_pos)

        info: dict = {"Reward": {}}
        info["actions"] = np.zeros((num, self._model.num_actuators), dtype=np.float32)
        info["last_actions"] = np.zeros((num, self._model.num_actuators), dtype=np.float32)
        info["Reward"] = {
            "hinge_velocity": np.zeros((num,), dtype=np.float32),
            "spin": np.zeros((num,), dtype=np.float32),
            "spin_sparse": np.zeros((num,), dtype=np.float32),
            "touch_raw": np.zeros((num,), dtype=np.float32),
            "touch_bonus": np.zeros((num,), dtype=np.float32),
            "approach_dist": np.zeros((num,), dtype=np.float32),
            "approach_reward": np.zeros((num,), dtype=np.float32),
        }

        obs = self._get_obs(data)
        return obs, info


@registry.env("dm-finger-turn-easy", "np")
@registry.env("dm-finger-turn-hard", "np")
class FingerTurnEnv(FingerEnv):
    def _init_obs_space(self):
        # Match dm_control's observation dict, but flatten into a vector.
        # Turn: position(4) + velocity(3) + touch(2) + target_position(2) + dist_to_target(1) = 12
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = data.dof_pos
        qvel = data.dof_vel
        position = np.concatenate(
            [
                qpos[:, [self._prox_qpos_i, self._dist_qpos_i]],
                self._tip_position_xz(data),
            ],
            axis=-1,
        )
        velocity = qvel[:, [self._prox_qvel_i, self._dist_qvel_i, self._hinge_qvel_i]]
        touch = self._touch(data)
        target_position = self._target_position_xz(data)
        dist_to_target = self._dist_to_target(data).reshape(data.shape[0], 1)
        return np.concatenate([position, velocity, touch, target_position, dist_to_target], axis=-1).astype(np.float32)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)
        terminated = np.isnan(obs).any(axis=-1)

        dist_to_target = self._dist_to_target(data)
        turn_sparse = (dist_to_target <= 0.0).astype(np.float32)

        touch_raw = np.zeros((data.shape[0],), dtype=np.float32)
        touch_bonus = np.zeros((data.shape[0],), dtype=np.float32)
        approach_dist = np.zeros((data.shape[0],), dtype=np.float32)
        approach_reward = np.zeros((data.shape[0],), dtype=np.float32)
        action_l2 = np.zeros((data.shape[0],), dtype=np.float32)
        action_delta_l2 = np.zeros((data.shape[0],), dtype=np.float32)

        if self._cfg.reward_mode == "shaped":
            # Encourage approaching the spinner so the agent actually makes contact and can rotate it.
            spinner_xyz = self._spinner.get_position(data)
            top_xyz = self._touchtop_site.get_position(data)
            bottom_xyz = self._touchbottom_site.get_position(data)
            top_dist = np.linalg.norm((top_xyz - spinner_xyz)[:, [0, 2]], axis=-1)
            bottom_dist = np.linalg.norm((bottom_xyz - spinner_xyz)[:, [0, 2]], axis=-1)
            approach_dist = np.minimum(top_dist, bottom_dist).astype(np.float32)
            sigma = max(float(self._cfg.turn_approach_sigma), 1e-6)
            approach_reward = (self._cfg.turn_approach_reward_scale * np.exp(-approach_dist / sigma)).astype(np.float32)

            dist_pos = np.maximum(dist_to_target, 0.0).astype(np.float32)
            if getattr(self._cfg, "turn_reward_shape", "linear") == "exp":
                sigma = float(
                    max(self._cfg.turn_reward_sigma_scale * self._target_radius, self._cfg.turn_reward_sigma_min)
                )
                sigma = max(sigma, 1e-6)
                turn = np.exp(-dist_pos / sigma).astype(np.float32)
            else:
                margin = float(
                    max(self._cfg.turn_reward_margin_scale * self._target_radius, self._cfg.turn_reward_min_margin)
                )
                margin = max(margin, 1e-6)
                # Dense reward: 1 inside target sphere, decays to 0 at `margin` outside.
                turn = np.clip(1.0 - dist_pos / margin, 0.0, 1.0).astype(np.float32)
            if self._cfg.turn_shaped_reward_beta != 1.0:
                turn = np.power(turn, self._cfg.turn_shaped_reward_beta, dtype=np.float32)

            # Encourage making contact (to actually be able to rotate the spinner)
            top = np.asarray(self._model.get_sensor_value("touchtop", data)).reshape(data.shape[0], -1)[:, 0]
            bottom = np.asarray(self._model.get_sensor_value("touchbottom", data)).reshape(data.shape[0], -1)[:, 0]
            touch_raw = (top + bottom).astype(np.float32)
            touch_bonus = self._cfg.turn_touch_bonus_scale * np.tanh(touch_raw / self._cfg.turn_touch_bonus_tanh_scale)

            # Reduce jitter: penalize large actions and action changes
            actions = state.info.get("actions", data.actuator_ctrls).astype(np.float32)
            last_actions = state.info.get("last_actions", actions).astype(np.float32)
            action_l2 = np.mean(np.square(actions), axis=-1).astype(np.float32)
            action_delta_l2 = np.mean(np.square(actions - last_actions), axis=-1).astype(np.float32)

            turn = (
                turn
                + approach_reward
                + touch_bonus
                - self._cfg.turn_action_l2_penalty_scale * action_l2
                - self._cfg.turn_action_delta_l2_penalty_scale * action_delta_l2
            ).astype(np.float32)
            turn = np.clip(turn, 0.0, 1.0).astype(np.float32)
        else:
            turn = turn_sparse

        rwd = turn
        state.info["Reward"] = {
            "dist_to_target": dist_to_target.copy(),
            "turn": turn.copy(),
            "turn_sparse": turn_sparse.copy(),
            "touch_raw": touch_raw.copy(),
            "touch_bonus": touch_bonus.copy(),
            "approach_dist": approach_dist.copy(),
            "approach_reward": approach_reward.copy(),
            "action_l2": action_l2.copy(),
            "action_delta_l2": action_delta_l2.copy(),
        }
        state.info["target_info"] = {"positions": self._target_xyz.copy(), "radius": self._target_radius}

        rwd[terminated] = 0.0
        return state.replace(obs=obs, reward=rwd, terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num = int(data.shape[0])
        dof_pos = np.zeros((num, self._model.num_dof_pos), dtype=np.float32)
        target_free_pos = self._maybe_init_target_freejoint(dof_pos)
        self._reset_collision_free_joint_angles(data, dof_pos, target_free_pos)

        hinge_xyz = self._spinner.get_position(data)
        # Match dm_control: radius = cap1.geom_size.sum() for capsule (radius + half-length).
        radius = float(np.sum(self._cap1.size[:2]))
        target_angle = np.random.uniform(-np.pi, np.pi, size=(num,))
        target_x = hinge_xyz[:, 0] + radius * np.sin(target_angle)
        target_z = hinge_xyz[:, 2] + radius * np.cos(target_angle)
        self._target_xyz = np.stack([target_x, hinge_xyz[:, 1], target_z], axis=-1).astype(np.float32)

        # Best-effort visualization when num_envs == 1 (site position is model-shared).
        if self._num_envs == 1:
            try:
                self._target_site.local_pos = self._target_xyz[0]
                self._target_site.size = np.asarray([self._target_radius], dtype=np.float32)
            except Exception:
                pass

        # If we have a freejoint-backed visual target (geom), set its pose in the state.
        if target_free_pos is not None:
            dof_pos[:, target_free_pos] = np.concatenate(
                [
                    self._target_xyz.astype(np.float32),
                    np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (num, 1)),
                ],
                axis=-1,
            )
            data.set_dof_pos(dof_pos, self._model)
            self._model.forward_kinematic(data)

        info: dict = {"Reward": {}}
        info["actions"] = np.zeros((num, self._model.num_actuators), dtype=np.float32)
        info["last_actions"] = np.zeros((num, self._model.num_actuators), dtype=np.float32)
        info["target_info"] = {"positions": self._target_xyz.copy(), "radius": self._target_radius}
        info["Reward"] = {
            "dist_to_target": np.zeros((num,), dtype=np.float32),
            "turn": np.zeros((num,), dtype=np.float32),
            "turn_sparse": np.zeros((num,), dtype=np.float32),
        }

        obs = self._get_obs(data)
        return obs, info
