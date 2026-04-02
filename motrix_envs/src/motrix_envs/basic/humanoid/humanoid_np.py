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
from motrix_envs.basic.humanoid.cfg import HumanoidWalkCfg
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("dm-humanoid-stand", "np")
@registry.env("dm-humanoid-walk", "np")
@registry.env("dm-humanoid-run", "np")
class Humanoid3DEnv(NpEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: HumanoidWalkCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_obs_space()
        self._init_action_space()

        self._torso = self._model.get_link("torso")
        self._head = self._model.get_link("head")
        self._pelvis = self._model.get_link("pelvis")
        self._left_hand = self._model.get_link("left_hand")
        self._right_hand = self._model.get_link("right_hand")
        self._left_foot = self._model.get_link("left_foot")
        self._right_foot = self._model.get_link("right_foot")

        self._move_speed = float(cfg.move_speed)
        self._stand_height = float(cfg.stand_height)
        self._target_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._target_direction_xy = self._target_direction[:2].copy()

        self._qpos_low, self._qpos_high = self._build_qpos_limits(self._model)
        self._cache_derived_constants(cfg)
        self._init_joint_randomization_config(cfg)

    def _build_qpos_limits(self, model) -> tuple[np.ndarray, np.ndarray]:
        num_dof_pos = int(model.num_dof_pos)
        jl = model.joint_limits
        if jl.ndim != 2 or jl.shape[0] != 2:
            low = np.full((num_dof_pos,), -np.inf, dtype=np.float32)
            high = np.full((num_dof_pos,), np.inf, dtype=np.float32)
            return low, high

        k = int(jl.shape[1])
        low = np.full((num_dof_pos,), -np.inf, dtype=np.float32)
        high = np.full((num_dof_pos,), np.inf, dtype=np.float32)
        m = min(k, num_dof_pos)
        low[:m] = jl[0, :m]
        high[:m] = jl[1, :m]
        return low, high

    def _cache_derived_constants(self, cfg: HumanoidWalkCfg) -> None:
        t_cfg = cfg.termination_config
        self._head_height_min = self._stand_height * 0.95
        self._pelvis_height_min = 0.6 * self._stand_height
        self._pelvis_height_margin = 0.6 * self._stand_height
        self._term_head_height_min = float(t_cfg.head_height_factor) * self._stand_height
        self._term_torso_upright_threshold = float(t_cfg.torso_upright_threshold)
        self._term_extreme_vel_threshold = float(t_cfg.extreme_vel_threshold)

    def _init_obs_space(self):
        model = self._model
        num_joint_angles = model.num_dof_pos - 7
        num_head_height = 1
        num_extremities = 12
        num_torso_vertical = 3
        num_com_vel = 3
        num_qvel = model.num_dof_vel
        num_target_local = 3
        num_obs = (
            num_joint_angles
            + num_head_height
            + num_extremities
            + num_torso_vertical
            + num_com_vel
            + num_qvel
            + num_target_local
        )
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        model = self._model
        self._action_space = gym.spaces.Box(
            model.actuator_ctrl_limits[0],
            model.actuator_ctrl_limits[1],
            (model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        state.data.actuator_ctrls = actions
        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def update_observation(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)
        return state.replace(obs=obs)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        head_height = self._get_head_height(data)
        torso_upright = self._get_torso_upright(data)
        terminated = self._compute_terminated(data, head_height, torso_upright)

        return state.replace(
            terminated=terminated,
        )

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        terminated = state.terminated
        head_height = self._get_head_height(data)
        pelvis_height = self._get_pelvis_height(data)
        torso_upright = self._get_torso_upright(data)
        rwd, reward_components = self._compute_reward(data, head_height, torso_upright, pelvis_height)
        rwd, reward_components = self._apply_termination_mask(terminated, rwd, reward_components)
        state.info["Reward"] = reward_components
        return state.replace(reward=rwd)

    def _apply_termination_mask(
        self,
        terminated: np.ndarray,
        rwd: np.ndarray,
        reward_components: dict,
    ) -> tuple[np.ndarray, dict]:
        rwd = np.where(terminated, 0.0, rwd).astype(np.float32)
        for k, v in reward_components.items():
            reward_components[k] = np.where(terminated, 0.0, v).astype(np.float32)
        return rwd, reward_components

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        self._randomize_joints_inplace(data)
        obs = self._get_obs(data)
        return obs, {}

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        joint_angles = data.dof_pos[:, 7:]
        head_height = self._get_head_height(data)[:, None]
        extremities = self._get_extremities(data)

        torso_rot = self._torso.get_rotation_mat(data)
        torso_vertical = torso_rot[:, 2, :]

        com_vel = self._model.get_sensor_value("torso_subtreelinvel", data)

        qvel = data.dof_vel
        target_direction_local = self._get_target_direction_local(data)

        obs = np.concatenate(
            [joint_angles, head_height, extremities, torso_vertical, com_vel, qvel, target_direction_local], axis=-1
        )
        return obs

    def _get_head_height(self, data: mtx.SceneData) -> np.ndarray:
        return self._head.get_position(data)[:, 2]

    def _get_pelvis_height(self, data: mtx.SceneData) -> np.ndarray:
        return self._pelvis.get_position(data)[:, 2]

    def _get_torso_upright(self, data: mtx.SceneData) -> np.ndarray:
        torso_rot = self._torso.get_rotation_mat(data)
        return torso_rot[:, 2, 2]

    def _get_extremities(self, data: mtx.SceneData) -> np.ndarray:
        torso_rot = self._torso.get_rotation_mat(data)
        torso_pos = self._torso.get_position(data)

        parts = [
            self._left_hand.get_position(data),
            self._left_foot.get_position(data),
            self._right_hand.get_position(data),
            self._right_foot.get_position(data),
        ]
        out = []

        for p in parts:
            torso_to_limb = p - torso_pos
            v_body = np.einsum("ni,nij->nj", torso_to_limb, torso_rot)
            out.append(v_body)

        return np.concatenate(out, axis=-1)

    def _get_target_direction_local(self, data: mtx.SceneData) -> np.ndarray:
        n = int(data.shape[0])
        torso_rot = self._torso.get_rotation_mat(data)
        target_world = np.ones((n, 3), dtype=np.float32) * self._target_direction[None, :]
        target_local = np.einsum("ni,nij->nj", target_world, torso_rot)
        return target_local

    def _compute_reward(
        self,
        data: mtx.SceneData,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
        pelvis_height: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        posture_reward = self._compute_posture_reward(head_height, torso_upright, pelvis_height)
        speed_reward, energy_reward = self._compute_speed_and_energy_reward(data)
        gait_reward = self._compute_gait_reward(data)

        rwd = (posture_reward * speed_reward * energy_reward * gait_reward).astype(np.float32)

        comps = {
            "energy": energy_reward.astype(np.float32),
            "speed": speed_reward.astype(np.float32),
            "posture": posture_reward.astype(np.float32),
            "gait": gait_reward.astype(np.float32),
        }
        return rwd, comps

    def _compute_posture_reward(
        self,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
        pelvis_height: np.ndarray,
    ) -> np.ndarray:
        stand_reward = reward.tolerance(
            head_height,
            bounds=(self._head_height_min, float("inf")),
            margin=0.5,
        ).flatten()

        upright_reward = reward.tolerance(
            torso_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=0.9,
        ).flatten()

        pelvis_height_reward = reward.tolerance(
            pelvis_height,
            bounds=(self._pelvis_height_min, float("inf")),
            sigmoid="linear",
            margin=self._pelvis_height_margin,
        ).flatten()

        return stand_reward * upright_reward * pelvis_height_reward

    def _compute_speed_and_energy_reward(
        self,
        data: mtx.SceneData,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_dir_xy = self._target_direction_xy

        ctrls = data.actuator_ctrls
        com_vel = self._model.get_sensor_value("torso_subtreelinvel", data)

        if self._move_speed <= 0.0:
            energy_reward = np.exp(-1.0 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.linalg.norm(com_vel[:, :2], axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, self._move_speed),
                margin=1.0,
                value_at_margin=0.01,
            ).flatten()
        elif self._move_speed <= 3.0:
            energy_reward = np.exp(-0.5 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.sum(com_vel[:, :2] * target_dir_xy, axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, self._move_speed),
                margin=self._move_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            ).flatten()
        else:
            energy_reward = np.exp(-0.3 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.sum(com_vel[:, :2] * target_dir_xy, axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            ).flatten()

        return speed_reward, energy_reward

    def _compute_heading_reward(
        self,
        forward_vec: np.ndarray,
        target_dir: np.ndarray,
        bounds,
        margin,
    ) -> np.ndarray:
        dot = np.sum(forward_vec * target_dir, axis=-1)
        return reward.tolerance(
            dot,
            bounds=bounds,
            margin=margin,
            value_at_margin=0.0,
            sigmoid="linear",
        ).flatten()

    def _compute_gait_reward(self, data: mtx.SceneData) -> np.ndarray:
        target_dir = self._target_direction

        torso_rot = self._torso.get_rotation_mat(data)
        head_rot = self._head.get_rotation_mat(data)
        pelvis_rot = self._pelvis.get_rotation_mat(data)

        torso_forward = torso_rot[:, 0, 0:3]
        torso_heading_reward = self._compute_heading_reward(torso_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        head_forward = head_rot[:, 0, 0:3]
        head_heading_reward = self._compute_heading_reward(head_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        pelvis_forward = pelvis_rot[:, 0, 0:3]
        pelvis_yaw_reward = self._compute_heading_reward(pelvis_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        pelvis_up = pelvis_rot[:, 2, 2]
        pelvis_level_reward = reward.tolerance(
            pelvis_up,
            bounds=(0.9, 1.0),
            margin=0.3,
            sigmoid="linear",
            value_at_margin=0.0,
        ).flatten()

        left_foot_pos = self._left_foot.get_position(data)
        right_foot_pos = self._right_foot.get_position(data)
        max_foot_h = np.maximum(left_foot_pos[:, 2], right_foot_pos[:, 2])
        feet_height_reward = reward.tolerance(
            max_foot_h,
            bounds=(0.0, 0.3),
            margin=0.5,
            sigmoid="quadratic",
            value_at_margin=0.0,
        ).flatten()

        return torso_heading_reward * head_heading_reward * pelvis_yaw_reward * pelvis_level_reward * feet_height_reward

    def _compute_terminated(
        self,
        data: mtx.SceneData,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
    ) -> np.ndarray:
        qpos = data.dof_pos
        qvel = data.dof_vel
        bad = ~np.isfinite(qpos).all(axis=-1) | ~np.isfinite(qvel).all(axis=-1)
        too_low = head_height < self._term_head_height_min
        too_tilted = torso_upright < self._term_torso_upright_threshold
        extreme_vel = np.abs(qvel).max(axis=-1) > self._term_extreme_vel_threshold
        return bad | too_low | too_tilted | extreme_vel

    def _init_joint_randomization_config(self, cfg: HumanoidWalkCfg) -> None:
        init_cfg = cfg.init_state
        self._reset_height = self._stand_height * init_cfg.reset_height_factor
        self._reset_qvel_range = init_cfg.reset_qvel_range
        self._reset_actuator_range = init_cfg.reset_actuator_range

        self._hip_yaw_range = tuple(np.deg2rad(x) for x in init_cfg.hip_yaw_range)
        self._hip_roll_range = tuple(np.deg2rad(x) for x in init_cfg.hip_roll_range)
        self._hip_pitch_range = tuple(np.deg2rad(x) for x in init_cfg.hip_pitch_range)

        self._symmetric_leg_pairs_rad = [
            (left_idx, right_idx, tuple(np.deg2rad(x) for x in deg_range))
            for left_idx, right_idx, deg_range in init_cfg.symmetric_leg_pairs
        ]
        self._symmetric_arm_pairs = init_cfg.symmetric_arm_pairs
        self._arm_margin_factor = init_cfg.arm_margin_factor
        self._symmetric_arm_used_indices = set()
        for left_idx, right_idx in self._symmetric_arm_pairs:
            self._symmetric_arm_used_indices.add(left_idx)
            self._symmetric_arm_used_indices.add(right_idx)

    def _randomize_joints_inplace(self, data: mtx.SceneData) -> None:
        # qpos layout (humanoid.xml): 0-6 free (x,y,z,qw,qx,qy,qz), 7=abdomen_z, 8=abdomen_y, 9=abdomen_x,
        # 10-15 right leg (hip_x,z,y, knee, ankle_y,x), 16-21 left leg, 22-24 right arm, 25-27 left arm (num_dof_pos=28)
        model = self._model
        n = int(data.shape[0])
        num_dof_pos = int(model.num_dof_pos)
        num_dof_vel = int(model.num_dof_vel)
        num_actuators = int(model.num_actuators)
        low, high = self._qpos_low, self._qpos_high

        qpos = np.zeros((n, num_dof_pos), dtype=np.float32)
        qpos[:, 2] = self._reset_height
        qpos[:, 3] = 1.0

        # qpos 7=abdomen_z (yaw), 8=abdomen_y (pitch), 9=abdomen_x (roll) per humanoid.xml
        qpos[:, 7] = np.random.uniform(self._hip_yaw_range[0], self._hip_yaw_range[1], size=(n,))
        qpos[:, 8] = np.random.uniform(self._hip_pitch_range[0], self._hip_pitch_range[1], size=(n,))
        qpos[:, 9] = np.random.uniform(self._hip_roll_range[0], self._hip_roll_range[1], size=(n,))

        self._randomize_symmetric_legs(qpos, n, num_dof_pos, low, high)
        self._randomize_symmetric_arms(qpos, n, num_dof_pos, low, high)
        self._randomize_remaining_joints(qpos, n, num_dof_pos, low, high)

        qvel = np.random.uniform(-self._reset_qvel_range, self._reset_qvel_range, size=(n, num_dof_vel)).astype(
            np.float32
        )
        actuator_ctrls = np.random.uniform(
            -self._reset_actuator_range, self._reset_actuator_range, size=(n, num_actuators)
        ).astype(np.float32)

        qpos_set = qpos.copy()
        qpos_set[:, 3:7] = np.concatenate([qpos[:, 4:7], qpos[:, 3:4]], axis=1)
        data.set_dof_pos(qpos_set, self._model)
        data.set_dof_vel(qvel)
        data.actuator_ctrls[:] = actuator_ctrls
        self._model.forward_kinematic(data)

    def _randomize_symmetric_legs(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        for left_idx, right_idx, (min_rad, max_rad) in self._symmetric_leg_pairs_rad:
            if left_idx < num_dof_pos:
                qpos[:, left_idx] = np.random.uniform(
                    np.clip(min_rad, low[left_idx], high[left_idx]),
                    np.clip(max_rad, low[left_idx], high[left_idx]),
                    size=(n,),
                )
            if right_idx < num_dof_pos:
                right_min_rad = -max_rad
                right_max_rad = -min_rad
                qpos[:, right_idx] = np.random.uniform(
                    np.clip(right_min_rad, low[right_idx], high[right_idx]),
                    np.clip(right_max_rad, low[right_idx], high[right_idx]),
                    size=(n,),
                )

    def _randomize_symmetric_arms(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        # Default range when model joint_limits are missing (low/high are ±inf);
        # np.random.uniform requires finite bounds.
        default_lo, default_hi = -np.pi, np.pi
        for left_idx, right_idx in self._symmetric_arm_pairs:
            if left_idx < num_dof_pos and right_idx < num_dof_pos:
                lo_l = low[left_idx] if np.isfinite(low[left_idx]) else default_lo
                hi_l = high[left_idx] if np.isfinite(high[left_idx]) else default_hi
                lo_r = low[right_idx] if np.isfinite(low[right_idx]) else default_lo
                hi_r = high[right_idx] if np.isfinite(high[right_idx]) else default_hi

                left_range = hi_l - lo_l
                left_margin = left_range * self._arm_margin_factor

                left_min = lo_l + left_margin
                left_max = hi_l - left_margin

                right_min = -left_max
                right_max = -left_min

                right_min_clipped = max(right_min, lo_r)
                right_max_clipped = min(right_max, hi_r)

                if left_min < left_max:
                    qpos[:, left_idx] = np.random.uniform(left_min, left_max, size=(n,))
                else:
                    qpos[:, left_idx] = np.random.uniform(lo_l, hi_l, size=(n,))

                if right_min_clipped < right_max_clipped:
                    qpos[:, right_idx] = np.random.uniform(right_min_clipped, right_max_clipped, size=(n,))
                else:
                    qpos[:, right_idx] = np.random.uniform(lo_r, hi_r, size=(n,))

    def _randomize_remaining_joints(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        used_indices = self._symmetric_arm_used_indices
        default_lo, default_hi = -np.pi, np.pi

        # 22 = first arm joint (right_shoulder1) in humanoid.xml qpos order;
        # arms 22-27 are covered by symmetric_arm_pairs
        for i in range(22, num_dof_pos):
            if i not in used_indices:
                lo = low[i] if np.isfinite(low[i]) else default_lo
                hi = high[i] if np.isfinite(high[i]) else default_hi
                qpos[:, i] = np.random.uniform(lo, hi, size=(n,))
