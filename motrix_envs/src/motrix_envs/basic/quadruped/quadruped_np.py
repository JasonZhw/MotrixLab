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
from motrix_envs.basic.quadruped.cfg import QuadrupedBaseCfg
from motrix_envs.math import quaternion
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState

_RANGEFINDER_SENSORS = [f"rf_{row}{col}" for row in range(4) for col in range(5)]


class QuadrupedEnv(NpEnv):
    _cfg: QuadrupedBaseCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: QuadrupedBaseCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._cfg = cfg
        self._torso = self._model.get_link("torso")
        self._floor_geom = self._model.get_geom("floor")

        self._workspace_site = None
        if cfg.include_origin:
            self._workspace_site = self._model.get_site("workspace")

        self._target_site = None
        if cfg.include_target:
            self._target_site = self._model.get_site("target")

        self._ball_body = None
        self._ball_geom = None
        if cfg.include_ball:
            self._ball_body = self._model.get_body("ball")
            self._ball_geom = self._model.get_geom("ball")

        self._leg_ball_geoms = []
        self._leg_ball_geom_count = 0
        self._leg_ball_geom_slices = []
        if cfg.include_ball:
            try:
                leg_body_geom_names = [
                    ["thigh_front_left", "shin_front_left", "foot_front_left", "toe_front_left"],
                    ["thigh_front_right", "shin_front_right", "foot_front_right", "toe_front_right"],
                    ["thigh_back_right", "shin_back_right", "foot_back_right", "toe_back_right"],
                    ["thigh_back_left", "shin_back_left", "foot_back_left", "toe_back_left"],
                ]
                leg_geoms = []
                start = 0
                for geom_names in leg_body_geom_names:
                    stop = start
                    for name in geom_names:
                        try:
                            geom = self._model.get_geom(name)
                        except Exception:
                            continue
                        leg_geoms.append(geom)
                        stop += 1
                    self._leg_ball_geom_slices.append(slice(start, stop))
                    start = stop
                if leg_geoms:
                    self._leg_ball_geoms = leg_geoms
                    self._leg_ball_geom_count = len(leg_geoms)
            except Exception:
                self._leg_ball_geom_slices = []

        self._body_dof_pos = self._model.num_dof_pos - 7 - (7 if cfg.include_ball else 0)
        self._body_dof_vel = self._model.num_dof_vel - 6 - (6 if cfg.include_ball else 0)
        self._dof_pos_slice = slice(7, 7 + self._body_dof_pos)
        self._dof_vel_slice = slice(6, 6 + self._body_dof_vel)
        self._ball_pos_slice = None
        self._ball_vel_slice = None
        if cfg.include_ball:
            self._ball_pos_slice = slice(self._model.num_dof_pos - 7, self._model.num_dof_pos)
            self._ball_vel_slice = slice(self._model.num_dof_vel - 6, self._model.num_dof_vel)

        self._init_dof_pos = self._model.compute_init_dof_pos().astype(np.float32)
        self._default_body_dof_pos = self._init_dof_pos[self._dof_pos_slice].copy()
        self._terrain_size = float(cfg.terrain_size)
        try:
            if self._model.num_hfields:
                hfield = self._model.get_hfield(0)
                self._terrain_size = max(self._terrain_size, float(abs(hfield.bound[3])))
        except Exception:
            pass

        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        num_obs = self._body_dof_pos + self._body_dof_vel + self._model.num_actuators
        num_obs += 3  # torso velocity
        num_obs += 1  # torso upright
        num_obs += 6  # imu accel + gyro

        if self._cfg.include_origin:
            num_obs += 3
        if self._cfg.include_rangefinder:
            num_obs += len(_RANGEFINDER_SENSORS)
        if self._cfg.include_ball:
            num_obs += 9
        if self._cfg.include_target:
            num_obs += 3

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

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        if self._cfg.clip_env_actions:
            actions = np.clip(actions, self._action_space.low, self._action_space.high)
        actions = actions.astype(np.float32)
        if "actions" not in state.info:
            state.info["actions"] = np.zeros_like(actions, dtype=np.float32)
        if "last_actions" not in state.info:
            state.info["last_actions"] = np.zeros_like(actions, dtype=np.float32)
        state.info["last_actions"] = state.info["actions"]
        state.info["actions"] = actions
        state.data.actuator_ctrls = actions
        return state

    def _sensor_value(self, data: mtx.SceneData, name: str) -> np.ndarray:
        value = np.asarray(self._model.get_sensor_value(name, data))
        return value.reshape(data.shape[0], -1)

    def _sensor_vector(self, data: mtx.SceneData, names: list[str]) -> np.ndarray:
        if not names:
            return np.zeros((data.shape[0], 0), dtype=np.float32)
        values = [self._sensor_value(data, name) for name in names]
        return np.concatenate(values, axis=-1)

    def _egocentric_state(self, data: mtx.SceneData) -> np.ndarray:
        dof_pos = data.dof_pos[:, self._dof_pos_slice]
        dof_vel = data.dof_vel[:, self._dof_vel_slice]
        act = data.actuator_ctrls
        return np.concatenate([dof_pos, dof_vel, act], axis=-1)

    def _torso_upright(self, data: mtx.SceneData) -> np.ndarray:
        return self._torso.get_rotation_mat(data)[:, 2, 2]

    def _torso_velocity(self, data: mtx.SceneData) -> np.ndarray:
        return self._sensor_value(data, "velocimeter")

    def _imu(self, data: mtx.SceneData) -> np.ndarray:
        accel = self._sensor_value(data, "imu_accel")
        gyro = self._sensor_value(data, "imu_gyro")
        return np.concatenate([accel, gyro], axis=-1)

    def _rangefinder(self, data: mtx.SceneData) -> np.ndarray:
        readings = self._sensor_vector(data, _RANGEFINDER_SENSORS)
        no_intersection = -1.0
        return np.where(readings == no_intersection, 1.0, np.tanh(readings))

    def _origin(self, data: mtx.SceneData) -> np.ndarray:
        torso_pos = self._torso.get_position(data)
        torso_frame = self._torso.get_rotation_mat(data)
        return -np.einsum("ni,nij->nj", torso_pos, torso_frame)

    def _origin_distance(self, data: mtx.SceneData) -> np.ndarray:
        workspace_pos = self._workspace_site.get_position(data)
        return np.linalg.norm(workspace_pos, axis=-1)

    def _ball_state(self, data: mtx.SceneData) -> np.ndarray:
        ball_pose = self._ball_body.get_pose(data)
        ball_pos = ball_pose[:, :3]
        torso_pos = self._torso.get_position(data)
        torso_frame = self._torso.get_rotation_mat(data)

        ball_rel_pos = ball_pos - torso_pos
        root_linvel = data.dof_vel[:, :3]
        ball_vel = data.dof_vel[:, self._ball_vel_slice]
        ball_rel_vel = ball_vel[:, :3] - root_linvel
        ball_rot_vel = ball_vel[:, 3:]

        stacked = np.stack([ball_rel_pos, ball_rel_vel, ball_rot_vel], axis=1)
        local = np.einsum("nij,njk->nik", stacked, torso_frame)
        return local.reshape(data.shape[0], -1)

    def _target_position(self, data: mtx.SceneData) -> np.ndarray:
        torso_pos = self._torso.get_position(data)
        torso_frame = self._torso.get_rotation_mat(data)
        to_target = self._target_site.get_position(data) - torso_pos
        return np.einsum("ni,nij->nj", to_target, torso_frame)

    def _ball_to_target_distance(self, data: mtx.SceneData) -> np.ndarray:
        ball_pos = self._ball_body.get_pose(data)[:, :3]
        target_pos = self._target_site.get_position(data)
        return np.linalg.norm((target_pos - ball_pos)[:, :2], axis=-1)

    def _aggregate_leg_ball_proximity(self, geom_penalties: np.ndarray) -> np.ndarray:
        num_legs = len(self._leg_ball_geom_slices)
        if num_legs == 0:
            return np.zeros((geom_penalties.shape[0], 0), dtype=np.float32)

        leg_penalties = []
        for geom_slice in self._leg_ball_geom_slices:
            if geom_slice.start == geom_slice.stop:
                leg_penalties.append(np.zeros((geom_penalties.shape[0],), dtype=np.float32))
            else:
                leg_penalties.append(geom_penalties[:, geom_slice].max(axis=-1))
        return np.stack(leg_penalties, axis=-1)

    def _point_to_segment_distance(self, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        segment = end - start
        segment_sq_norm = np.sum(segment * segment, axis=-1)
        safe_norm = np.where(segment_sq_norm > 1e-8, segment_sq_norm, 1.0)
        t = np.sum((point - start) * segment, axis=-1) / safe_norm
        t = np.where(segment_sq_norm > 1e-8, np.clip(t, 0.0, 1.0), 0.0)
        closest = start + t[:, None] * segment
        return np.linalg.norm(point - closest, axis=-1)

    def _geom_ball_surface_clearance(
        self, geom: mtx.Geom, ball_pos: np.ndarray, ball_radius: float, data: mtx.SceneData
    ) -> np.ndarray:
        geom_pose = geom.get_pose(data)
        geom_pos = geom_pose[:, :3]
        geom_quat = geom_pose[:, 3:]
        geom_size = np.atleast_1d(np.asarray(geom.size, dtype=np.float32))
        geom_radius = float(geom_size[0])

        if getattr(geom, "shape", None) == mtx.Shape.Capsule and geom_size.shape[0] > 1 and geom_size[1] > 0.0:
            half_length = float(geom_size[1])
            axis = quaternion.rotate_vector(geom_quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            start = geom_pos - axis * half_length
            end = geom_pos + axis * half_length
            center_distance = self._point_to_segment_distance(ball_pos, start, end)
        else:
            center_distance = np.linalg.norm(ball_pos - geom_pos, axis=-1)

        return center_distance - (ball_radius + geom_radius)

    def _leg_body_ball_penalty(self, data: mtx.SceneData) -> np.ndarray:
        num_legs = len(self._leg_ball_geom_slices)
        if self._ball_geom is None or self._leg_ball_geom_count == 0 or num_legs == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)

        ball_pos = self._ball_geom.get_pose(data)[:, :3]
        ball_radius = float(np.atleast_1d(self._ball_geom.size)[0])
        geom_penalties = []
        for geom in self._leg_ball_geoms:
            clearance = self._geom_ball_surface_clearance(geom, ball_pos, ball_radius, data)
            geom_penalties.append(
                reward.tolerance(
                    -clearance,
                    bounds=(0.0, float("inf")),
                    margin=self._cfg.fetch_leg_ball_penalty_margin,
                    value_at_margin=0.0,
                    sigmoid="linear",
                )
            )

        proximity = np.stack(geom_penalties, axis=-1).astype(np.float32)
        leg_penalties = self._aggregate_leg_ball_proximity(proximity)
        return leg_penalties.sum(axis=-1).astype(np.float32)

    def _fetch_stability_gate(self, torso_upright: np.ndarray, torso_height: np.ndarray) -> np.ndarray:
        upright_gate = reward.tolerance(
            torso_upright,
            bounds=(self._cfg.fetch_stability_upright_min, float("inf")),
            margin=self._cfg.fetch_stability_upright_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        height_gate = reward.tolerance(
            torso_height,
            bounds=(self._cfg.fetch_stability_height_min, float("inf")),
            margin=self._cfg.fetch_stability_height_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return upright_gate * height_gate

    def _fetch_fall_terminated(self, torso_upright: np.ndarray, torso_height: np.ndarray) -> np.ndarray:
        return (torso_upright < self._cfg.fetch_fall_upright_min) | (torso_height < self._cfg.fetch_fall_height_min)

    def _upright_reward(self, torso_upright: np.ndarray) -> np.ndarray:
        deviation = float(np.cos(np.deg2rad(self._cfg.deviation_angle)))
        return reward.tolerance(
            torso_upright,
            bounds=(deviation, float("inf")),
            margin=1 + deviation,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _move_reward(self, torso_vel: np.ndarray) -> np.ndarray:
        return reward.tolerance(
            torso_vel[:, 0],
            bounds=(self._cfg.desired_speed, float("inf")),
            margin=self._cfg.desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    def _backward_penalty(self, torso_vel: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, -torso_vel[:, 0])

    def _escape_reward(self, data: mtx.SceneData) -> np.ndarray:
        return reward.tolerance(
            self._origin_distance(data),
            bounds=(self._terrain_size, float("inf")),
            margin=self._terrain_size,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _radial_speed_reward(self, data: mtx.SceneData) -> np.ndarray:
        radial_speed_reward = np.zeros((data.shape[0],), dtype=np.float32)
        if not self._cfg.include_origin:
            return radial_speed_reward

        torso_pos = self._torso.get_position(data)
        radial_vec = torso_pos[:, :2]
        radial_norm = np.linalg.norm(radial_vec, axis=-1, keepdims=True)
        radial_dir = np.divide(radial_vec, radial_norm, out=np.zeros_like(radial_vec), where=radial_norm > 1e-6)
        radial_speed = np.sum(data.dof_vel[:, :2] * radial_dir, axis=-1)
        radial_speed = np.maximum(0.0, radial_speed)
        return reward.tolerance(
            radial_speed,
            bounds=(self._cfg.desired_speed, float("inf")),
            margin=self._cfg.desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    def _heading_reward(self, data: mtx.SceneData) -> np.ndarray:
        heading_reward = np.zeros((data.shape[0],), dtype=np.float32)
        if self._cfg.heading_reward_weight <= 0.0:
            return heading_reward

        torso_frame = self._torso.get_rotation_mat(data)
        heading_xy = torso_frame[:, 0, :2]
        heading_norm = np.linalg.norm(heading_xy, axis=-1, keepdims=True)
        heading_dir = np.divide(heading_xy, heading_norm, out=np.zeros_like(heading_xy), where=heading_norm > 1e-6)
        heading_align = heading_dir[:, 0]
        return reward.tolerance(
            heading_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.heading_reward_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _height_reward(self, data: mtx.SceneData) -> np.ndarray:
        torso_height = self._torso.get_position(data)[:, 2]
        return reward.tolerance(
            torso_height,
            bounds=(self._cfg.stand_height, float("inf")),
            margin=self._cfg.stand_height_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _lateral_reward(self, torso_vel: np.ndarray) -> np.ndarray:
        return reward.tolerance(
            np.abs(torso_vel[:, 1]),
            bounds=(0.0, self._cfg.lateral_velocity_limit),
            margin=self._cfg.lateral_velocity_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _smooth_reward(self, state: NpEnvState) -> np.ndarray:
        smooth_reward = np.zeros((state.data.shape[0],), dtype=np.float32)
        if "actions" not in state.info or "last_actions" not in state.info:
            return smooth_reward

        delta = state.info["actions"] - state.info["last_actions"]
        delta_norm = np.linalg.norm(delta, axis=-1)
        return reward.tolerance(
            delta_norm,
            bounds=(0.0, 0.0),
            margin=self._cfg.action_smoothness_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _lin_vel_z_penalty(self, torso_vel: np.ndarray) -> np.ndarray:
        return np.square(torso_vel[:, 2]).astype(np.float32)

    def _ang_vel_xy_penalty(self, data: mtx.SceneData) -> np.ndarray:
        imu = self._imu(data)
        return np.sum(np.square(imu[:, 3:5]), axis=1).astype(np.float32)

    def _similar_to_default_penalty(self, data: mtx.SceneData) -> np.ndarray:
        body_dof_pos = data.dof_pos[:, self._dof_pos_slice]
        return np.sum(np.abs(body_dof_pos - self._default_body_dof_pos), axis=1).astype(np.float32)

    def _locomotion_reward_terms(
        self,
        upright_reward: np.ndarray,
        move_reward: np.ndarray,
        backward_penalty: np.ndarray,
        height_reward: np.ndarray,
        lateral_reward: np.ndarray,
        heading_reward: np.ndarray,
        smooth_reward: np.ndarray,
        lin_vel_z_penalty: np.ndarray,
        ang_vel_xy_penalty: np.ndarray,
        similar_to_default_penalty: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "move": upright_reward * move_reward,
            "backward": backward_penalty,
            "height": height_reward,
            "lateral": lateral_reward,
            "heading": heading_reward,
            "smooth": smooth_reward,
            "lin_vel_z": lin_vel_z_penalty,
            "ang_vel_xy": ang_vel_xy_penalty,
            "similar_to_default": similar_to_default_penalty,
        }

    def _locomotion_reward_scales(self) -> dict[str, float]:
        return {
            "move": 1.0,
            "backward": -self._cfg.backward_penalty_weight,
            "height": self._cfg.height_reward_weight,
            "lateral": self._cfg.lateral_reward_weight,
            "heading": self._cfg.heading_reward_weight,
            "smooth": self._cfg.action_smoothness_weight,
            "lin_vel_z": -self._cfg.lin_vel_z_weight,
            "ang_vel_xy": -self._cfg.ang_vel_xy_weight,
            "similar_to_default": -self._cfg.similar_to_default_weight,
        }

    def _escape_reward_terms(
        self, upright_reward: np.ndarray, escape_reward: np.ndarray, radial_speed_reward: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {
            "escape": upright_reward * escape_reward,
            "radial": radial_speed_reward,
        }

    def _escape_reward_scales(self) -> dict[str, float]:
        return {
            "escape": 1.0,
            "radial": self._cfg.radial_velocity_weight,
        }

    def _sum_scaled_rewards(self, reward_terms: dict[str, np.ndarray], reward_scales: dict[str, float]) -> np.ndarray:
        rewards = {name: value * reward_scales[name] for name, value in reward_terms.items()}
        return sum(rewards.values())

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        parts = [
            self._egocentric_state(data),
            self._torso_velocity(data),
            self._torso_upright(data).reshape(data.shape[0], 1),
            self._imu(data),
        ]

        if self._cfg.include_origin:
            parts.append(self._origin(data))
        if self._cfg.include_rangefinder:
            parts.append(self._rangefinder(data))
        if self._cfg.include_ball:
            parts.append(self._ball_state(data))
        if self._cfg.include_target:
            parts.append(self._target_position(data))

        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _locomotion_reward_info(self, num_envs: int) -> dict:
        return {
            "upright": np.zeros((num_envs,), dtype=np.float32),
            "move": np.zeros((num_envs,), dtype=np.float32),
            "backward": np.zeros((num_envs,), dtype=np.float32),
            "height": np.zeros((num_envs,), dtype=np.float32),
            "lateral": np.zeros((num_envs,), dtype=np.float32),
            "heading": np.zeros((num_envs,), dtype=np.float32),
            "smooth": np.zeros((num_envs,), dtype=np.float32),
            "lin_vel_z": np.zeros((num_envs,), dtype=np.float32),
            "ang_vel_xy": np.zeros((num_envs,), dtype=np.float32),
            "similar_to_default": np.zeros((num_envs,), dtype=np.float32),
            "total": np.zeros((num_envs,), dtype=np.float32),
        }

    def _escape_reward_info(self, num_envs: int) -> dict:
        info = self._locomotion_reward_info(num_envs)
        info.update(
            {
                "escape": np.zeros((num_envs,), dtype=np.float32),
                "radial": np.zeros((num_envs,), dtype=np.float32),
            }
        )
        return info

    def _fetch_reward_info(self, num_envs: int) -> dict:
        return {
            "upright": np.zeros((num_envs,), dtype=np.float32),
            "stage_move": np.zeros((num_envs,), dtype=np.float32),
            "stage_reach": np.zeros((num_envs,), dtype=np.float32),
            "stability": np.zeros((num_envs,), dtype=np.float32),
            "behind_align": np.zeros((num_envs,), dtype=np.float32),
            "face_ball": np.zeros((num_envs,), dtype=np.float32),
            "near_ball": np.zeros((num_envs,), dtype=np.float32),
            "ready": np.zeros((num_envs,), dtype=np.float32),
            "ready_gate": np.zeros((num_envs,), dtype=np.float32),
            "fetch": np.zeros((num_envs,), dtype=np.float32),
            "push": np.zeros((num_envs,), dtype=np.float32),
            "away": np.zeros((num_envs,), dtype=np.float32),
            "leg_ball": np.zeros((num_envs,), dtype=np.float32),
            "backward": np.zeros((num_envs,), dtype=np.float32),
            "total": np.zeros((num_envs,), dtype=np.float32),
        }

    def _base_locomotion_components(self, data: mtx.SceneData, state: NpEnvState) -> dict[str, np.ndarray]:
        torso_vel = self._torso_velocity(data)
        return {
            "move": self._move_reward(torso_vel),
            "backward": self._backward_penalty(torso_vel),
            "height": self._height_reward(data),
            "lateral": self._lateral_reward(torso_vel),
            "heading": self._heading_reward(data),
            "smooth": self._smooth_reward(state),
            "lin_vel_z": self._lin_vel_z_penalty(torso_vel),
            "ang_vel_xy": self._ang_vel_xy_penalty(data),
            "similar_to_default": self._similar_to_default_penalty(data),
        }

    def _locomotion_reward(self, upright_reward: np.ndarray, components: dict[str, np.ndarray]) -> np.ndarray:
        reward_terms = self._locomotion_reward_terms(
            upright_reward,
            components["move"],
            components["backward"],
            components["height"],
            components["lateral"],
            components["heading"],
            components["smooth"],
            components["lin_vel_z"],
            components["ang_vel_xy"],
            components["similar_to_default"],
        )
        return self._sum_scaled_rewards(reward_terms, self._locomotion_reward_scales())

    def _build_reset_info(self, num_envs: int) -> dict:
        return {
            "Reward": self._init_reward_info(num_envs),
            "actions": np.zeros((num_envs, self._model.num_actuators), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._model.num_actuators), dtype=np.float32),
        }

    def _random_quaternion(self, num: int) -> np.ndarray:
        q = np.random.randn(num, 4).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return q

    def _yaw_quaternion(self, yaw: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(yaw)
        half = yaw * 0.5
        return np.stack([zeros, zeros, np.sin(half), np.cos(half)], axis=-1).astype(np.float32)

    def _lift_non_contacting(self, data: mtx.SceneData, dof_pos: np.ndarray) -> np.ndarray:
        z = dof_pos[:, 2].copy()
        pending = np.ones((data.shape[0],), dtype=bool)
        for _ in range(1000):
            if not pending.any():
                break
            dof_pos[pending, 2] = z[pending]
            data.set_dof_pos(dof_pos, self._model)
            self._model.forward_kinematic(data)
            num_contacts = self._model.get_contact_query(data).num_contacts
            pending = num_contacts > 0
            z[pending] += 0.01
        return dof_pos

    def _finish_reset(self, data: mtx.SceneData, dof_pos: np.ndarray, dof_vel: np.ndarray) -> tuple[np.ndarray, dict]:
        dof_pos = self._lift_non_contacting(data, dof_pos)
        data.set_dof_pos(dof_pos, self._model)
        data.set_dof_vel(dof_vel)
        self._model.forward_kinematic(data)

        obs = self._get_obs(data)
        info = self._build_reset_info(int(data.shape[0]))
        return obs, info


@registry.env("dm-quadruped-walk", "np")
@registry.env("dm-quadruped-run", "np")
class QuadrupedLocomotionEnv(QuadrupedEnv):
    def _init_reward_info(self, num_envs: int) -> dict:
        return self._locomotion_reward_info(num_envs)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)

        torso_upright = self._torso_upright(data)
        upright_reward = self._upright_reward(torso_upright)
        locomotion_components = self._base_locomotion_components(data, state)
        rwd = self._locomotion_reward(upright_reward, locomotion_components)

        reward_components = {"upright": upright_reward}
        reward_components.update(locomotion_components)
        reward_components["total"] = rwd

        terminated = np.isnan(obs).any(axis=-1)
        rwd = np.where(terminated, 0.0, rwd).astype(np.float32)
        state.info["Reward"] = reward_components

        return state.replace(obs=obs, reward=rwd, terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num = int(data.shape[0])
        dof_pos = np.tile(self._init_dof_pos, (num, 1))
        dof_vel = np.zeros((num, self._model.num_dof_vel), dtype=np.float32)

        if self._cfg.fix_heading:
            dof_pos[:, 3:7] = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num, 1))
        else:
            dof_pos[:, 3:7] = self._random_quaternion(num)

        return self._finish_reset(data, dof_pos, dof_vel)


@registry.env("dm-quadruped-escape", "np")
class QuadrupedEscapeEnv(QuadrupedLocomotionEnv):
    def _init_reward_info(self, num_envs: int) -> dict:
        return self._escape_reward_info(num_envs)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)

        torso_upright = self._torso_upright(data)
        upright_reward = self._upright_reward(torso_upright)
        locomotion_components = self._base_locomotion_components(data, state)
        escape_reward = self._escape_reward(data)
        radial_speed_reward = self._radial_speed_reward(data)

        reward_terms = self._locomotion_reward_terms(
            upright_reward,
            locomotion_components["move"],
            locomotion_components["backward"],
            locomotion_components["height"],
            locomotion_components["lateral"],
            locomotion_components["heading"],
            locomotion_components["smooth"],
            locomotion_components["lin_vel_z"],
            locomotion_components["ang_vel_xy"],
            locomotion_components["similar_to_default"],
        )
        reward_scales = self._locomotion_reward_scales()
        reward_terms.update(self._escape_reward_terms(upright_reward, escape_reward, radial_speed_reward))
        reward_scales.update(self._escape_reward_scales())
        rwd = self._sum_scaled_rewards(reward_terms, reward_scales)

        reward_components = {"upright": upright_reward}
        reward_components.update(locomotion_components)
        reward_components.update(
            {
                "escape": escape_reward,
                "radial": radial_speed_reward,
                "total": rwd,
            }
        )

        terminated = np.isnan(obs).any(axis=-1)
        rwd = np.where(terminated, 0.0, rwd).astype(np.float32)
        state.info["Reward"] = reward_components

        return state.replace(obs=obs, reward=rwd, terminated=terminated)


@registry.env("dm-quadruped-fetch", "np")
class QuadrupedFetchEnv(QuadrupedEnv):
    def _init_reward_info(self, num_envs: int) -> dict:
        return self._fetch_reward_info(num_envs)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        obs = self._get_obs(data)

        torso_upright = self._torso_upright(data)
        upright_reward = self._upright_reward(torso_upright)
        torso_height = self._torso.get_position(data)[:, 2]
        stability_gate = self._fetch_stability_gate(torso_upright, torso_height)
        target_radius = float(self._cfg.target_radius)
        if self._target_site is not None:
            try:
                target_radius = float(np.atleast_1d(self._target_site.size)[0])
            except Exception:
                pass

        ball_pos = self._ball_body.get_pose(data)[:, :3]
        target_pos = self._target_site.get_position(data)
        torso_pos = self._torso.get_position(data)
        to_target = target_pos[:, :2] - ball_pos[:, :2]
        to_target_norm = np.linalg.norm(to_target, axis=-1, keepdims=True)
        to_target_dir = np.where(to_target_norm > 1e-6, to_target / to_target_norm, 0.0)
        torso_frame = self._torso.get_rotation_mat(data)
        heading_xy = torso_frame[:, 0, :2]
        heading_norm = np.linalg.norm(heading_xy, axis=-1, keepdims=True)
        heading_dir = np.where(heading_norm > 1e-6, heading_xy / heading_norm, 0.0)
        to_ball = ball_pos[:, :2] - torso_pos[:, :2]
        to_ball_norm = np.linalg.norm(to_ball, axis=-1, keepdims=True)
        to_ball_dir = np.where(to_ball_norm > 1e-6, to_ball / to_ball_norm, 0.0)

        behind_align = np.sum(to_ball_dir * to_target_dir, axis=-1)
        behind_align_reward = reward.tolerance(
            behind_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.fetch_behind_align_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        heading_align = np.sum(heading_dir * to_ball_dir, axis=-1)
        face_ball_reward = reward.tolerance(
            heading_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.fetch_heading_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_to_robot = torso_pos[:, :2] - ball_pos[:, :2]
        back_dir = -to_target_dir
        corridor_lat = np.linalg.norm(
            ball_to_robot - np.sum(ball_to_robot * back_dir, axis=-1, keepdims=True) * back_dir,
            axis=-1,
        )
        corridor_reward = reward.tolerance(
            corridor_lat,
            bounds=(0.0, self._cfg.fetch_corridor_width),
            margin=self._cfg.fetch_corridor_width,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_dist = np.linalg.norm(to_ball, axis=-1)
        near_ball_reward = reward.tolerance(
            ball_dist,
            bounds=(0.0, self._cfg.fetch_ready_ball_distance),
            margin=self._cfg.fetch_ready_ball_distance,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ready = behind_align_reward * face_ball_reward * corridor_reward * near_ball_reward
        ready_gate = reward.tolerance(
            ready,
            bounds=(self._cfg.fetch_ready_threshold, 1.0),
            margin=1.0 - self._cfg.fetch_ready_threshold,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        behind_pos = ball_pos[:, :2] + back_dir * self._cfg.fetch_behind_distance
        ahead_pos = ball_pos[:, :2] + to_target_dir * self._cfg.fetch_ahead_distance
        stage_pos = (1.0 - ready_gate)[:, None] * behind_pos + ready_gate[:, None] * ahead_pos
        if self._cfg.fetch_side_stage_offset > 0.0:
            side_dir = np.stack([-back_dir[:, 1], back_dir[:, 0]], axis=-1)
            ball_to_robot_side = np.sum(ball_to_robot * side_dir, axis=-1)
            side_sign = np.where(ball_to_robot_side >= 0.0, 1.0, -1.0)
            side_pos = behind_pos + (side_sign[:, None] * side_dir * self._cfg.fetch_side_stage_offset)
            use_side_stage = (
                (ready_gate < self._cfg.fetch_side_stage_gate_threshold)
                & (ball_dist < self._cfg.fetch_side_stage_ball_distance)
                & (behind_align < self._cfg.fetch_side_stage_align_threshold)
            )
            stage_pos = np.where(use_side_stage[:, None], side_pos, stage_pos)

        to_stage = stage_pos - torso_pos[:, :2]
        stage_dist = np.linalg.norm(to_stage, axis=-1)
        stage_dir = np.where(stage_dist[:, None] > 1e-6, to_stage / stage_dist[:, None], 0.0)

        speed_to_stage = np.sum(data.dof_vel[:, :2] * stage_dir, axis=-1)
        stage_move = reward.tolerance(
            speed_to_stage,
            bounds=(self._cfg.fetch_stage_speed, float("inf")),
            margin=self._cfg.fetch_stage_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        backward_penalty = np.maximum(0.0, -speed_to_stage)
        stage_reach = reward.tolerance(
            stage_dist,
            bounds=(0.0, self._cfg.fetch_stage_radius),
            margin=self._cfg.fetch_stage_radius,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        fetch_reward = reward.tolerance(
            self._ball_to_target_distance(data),
            bounds=(0.0, target_radius),
            margin=self._cfg.fetch_reward_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_vel = data.dof_vel[:, self._ball_vel_slice][:, :2]
        ball_speed_to_target = np.sum(ball_vel * to_target_dir, axis=-1)
        push_reward = reward.tolerance(
            np.maximum(0.0, ball_speed_to_target),
            bounds=(self._cfg.fetch_push_speed, float("inf")),
            margin=self._cfg.fetch_push_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        away_penalty = (1.0 - ready_gate) * np.maximum(0.0, -ball_speed_to_target)
        leg_ball_penalty = self._leg_body_ball_penalty(data)

        rwd = stability_gate * upright_reward * stage_move
        rwd -= self._cfg.fetch_backward_penalty_weight * backward_penalty
        rwd += self._cfg.fetch_stage_reward_weight * (stability_gate * stage_reach)
        rwd += self._cfg.fetch_heading_weight * (stability_gate * face_ball_reward)
        rwd += self._cfg.fetch_ready_weight * (stability_gate * ready)
        rwd += self._cfg.fetch_reward_weight * (stability_gate * ready_gate * fetch_reward)
        rwd += self._cfg.fetch_push_reward_weight * (stability_gate * ready_gate * push_reward)
        rwd -= self._cfg.fetch_away_penalty_weight * away_penalty
        rwd -= self._cfg.fetch_leg_ball_penalty_weight * leg_ball_penalty

        reward_components = {
            "upright": upright_reward,
            "stage_move": stage_move,
            "stage_reach": stage_reach,
            "stability": stability_gate,
            "behind_align": behind_align_reward,
            "face_ball": face_ball_reward,
            "near_ball": near_ball_reward,
            "ready": ready,
            "ready_gate": ready_gate,
            "fetch": fetch_reward,
            "push": push_reward,
            "away": away_penalty,
            "leg_ball": leg_ball_penalty,
            "backward": backward_penalty,
            "total": rwd,
        }

        terminated = np.isnan(obs).any(axis=-1)
        terminated |= self._fetch_fall_terminated(torso_upright, torso_height)
        rwd = np.where(terminated, 0.0, rwd).astype(np.float32)
        for key, value in reward_components.items():
            reward_components[key] = np.where(terminated, 0.0, value).astype(np.float32)
        state.info["Reward"] = reward_components

        return state.replace(obs=obs, reward=rwd, terminated=terminated)

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num = int(data.shape[0])
        dof_pos = np.tile(self._init_dof_pos, (num, 1))
        dof_vel = np.zeros((num, self._model.num_dof_vel), dtype=np.float32)

        floor_radius = float(self._floor_geom.size[0])
        if floor_radius <= 0.0:
            floor_radius = self._terrain_size
        spawn_radius = 0.12 * floor_radius
        yaw = np.random.uniform(0.0, 2 * np.pi, size=(num,))
        dof_pos[:, 0] = np.random.uniform(-spawn_radius, spawn_radius, size=(num,))
        dof_pos[:, 1] = np.random.uniform(-spawn_radius, spawn_radius, size=(num,))
        dof_pos[:, 3:7] = self._yaw_quaternion(yaw)

        ball_xy = np.random.uniform(-spawn_radius, spawn_radius, size=(num, 2))
        ball_qpos = self._ball_pos_slice
        dof_pos[:, ball_qpos.start : ball_qpos.start + 2] = ball_xy
        ball_radius = float(self._ball_geom.size[0]) if self._ball_geom is not None else 0.15
        dof_pos[:, ball_qpos.start + 2] = ball_radius
        dof_pos[:, ball_qpos.start + 3 : ball_qpos.stop] = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num, 1)
        )

        ball_qvel = self._ball_vel_slice
        dof_vel[:, ball_qvel.start : ball_qvel.stop] = 0.0

        return self._finish_reset(data, dof_pos, dof_vel)
