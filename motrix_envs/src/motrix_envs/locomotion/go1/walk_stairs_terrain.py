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
from motrix_envs.locomotion.go1.cfg import Go1WalkNpStairsEnvCfg
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .common import generate_repeating_array


@registry.env("go1-stairs-terrain-walk", sim_backend="np")
class Go1WalkStairsTask(NpEnv):
    _init_dof_pos: np.ndarray
    _init_dof_vel: np.ndarray

    def __init__(self, cfg: Go1WalkNpStairsEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()
        self._body = self._model.get_body(self.cfg.asset.body_name)
        self._num_action = self._action_space.shape[0]
        self._num_observation = self._observation_space.shape[0]
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=np.float32,
        )
        height_list = np.array([-1, 0.5, 1.5])
        offset_h = [[2, 0, 2, 1, 1], [2, 2, 1, 0, 0], [1, 1, 2, 1, 2], [0, 1, 0, 2, 0], [0, 1, 1, 0, 2]]
        offset = []
        for i in range(5):
            for j in range(5):
                h_index = offset_h[j][i]
                offset.append([(i - 2) * 8.0, (j - 2) * 8.0, height_list[h_index]])
        self.offset_list = np.array(offset)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_buffer()
        self.period_counter = 0

    def _init_obs_space(self):
        model = self.model
        num_dof_vel = model.num_dof_vel  # linvel + gyro + joint_vel
        num_joint_angle = model.num_dof_pos - 7
        num_gravity = 3
        num_actions = model.num_actuators
        num_command = 3
        num_contact_force = 12

        num_obs = num_dof_vel + num_joint_angle + num_gravity + num_actions + num_command + num_contact_force
        assert num_obs == 60

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        model = self.model
        self._action_space = gym.spaces.Box(
            np.array(model.actuator_ctrl_limits[0, :]),
            np.array(model.actuator_ctrl_limits[1, :]),
            (model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_vel(data)

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, Go1WalkNpStairsEnvCfg)
        # init buffers

        self.reset_buf = np.ones(self._num_envs, dtype=np.bool)
        self.kps = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.stiffness
        self.kds = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.damping
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            (
                [
                    cfg.normalization.lin_vel,
                    cfg.normalization.lin_vel,
                    cfg.normalization.ang_vel,
                ]
            ),
            dtype=np.float32,
        )

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.hip_indices = []
        self.calf_indices = []
        for i in range(self._model.num_actuators):
            for name in cfg.init_state.default_joint_angles.keys():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = cfg.init_state.default_joint_angles[name]
            if "hip" in self._model.actuator_names[i]:
                self.hip_indices.append(i)
            if "calf" in self._model.actuator_names[i]:
                self.calf_indices.append(i)

        self._init_dof_pos[-self._num_action :] = self.default_angles

        self.ground = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and cfg.asset.ground_name in geom_name:
                self.ground.append(self._model.get_geom_index(geom_name))
        self.termination_contact = None
        self.foot = []
        for gournd_index in self.ground:
            for name in cfg.asset.terminate_after_contacts_on:
                if self.termination_contact is None:
                    self.termination_contact = np.array(
                        [[self._model.get_geom_index(name), gournd_index]], dtype=np.uint32
                    )
                else:
                    self.termination_contact = np.append(
                        self.termination_contact,
                        np.array(
                            [[self._model.get_geom_index(name), gournd_index]],
                            dtype=np.uint32,
                        ),
                        axis=0,
                    )
        self.num_check = self.termination_contact.shape[0]

        self.foot = None
        for gournd_index in self.ground:
            for i in self._model.geom_names:
                if i is not None and cfg.asset.foot_name in i:
                    if self.foot is None:
                        self.foot = np.array([[self._model.get_geom_index(i), gournd_index]], dtype=np.uint32)
                    else:
                        self.foot = np.append(
                            self.foot,
                            np.array(
                                [[self._model.get_geom_index(i), gournd_index]],
                                dtype=np.uint32,
                            ),
                            axis=0,
                        )
        self.foot_check_num = self.foot.shape[0]
        self.foot_check = self.foot

        self.termination_check = self.termination_contact

        spacing = 2.0
        cols = int(np.ceil(np.sqrt(self._num_envs)))
        offsets = []
        for i in range(self._num_envs):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])
        self.offsets = np.array(offsets)

    def apply_action(self, actions, state):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions, data):
        # Compute torques from actions.
        # pd controller
        actions_scaled = actions * self.cfg.control_config.action_scale
        torques = self.kps * (
            actions_scaled + self.default_angles - self.get_dof_pos(data)
        ) - self.kds * self.get_dof_vel(data)
        return torques

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.local_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.gyro, data)

    def update_state(self, state):
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        diff = self.get_dof_pos(data) - self.default_angles
        noisy_linvel = linear_vel * self.cfg.normalization.lin_vel
        noisy_gyro = gyro * self.cfg.normalization.ang_vel
        noisy_joint_angle = diff * self.cfg.normalization.dof_pos
        noisy_joint_vel = self.get_dof_vel(data) * self.cfg.normalization.dof_vel
        command = info["commands"] * self.commands_scale
        last_actions = info["current_actions"]
        contact_force = info["contact_force"]

        obs = np.hstack(
            [
                noisy_linvel,
                noisy_gyro,
                local_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command,
                contact_force,
            ]
        )
        return obs

    def update_observation(self, state: NpEnvState):
        data = state.data
        # self.border_check(data, state.info)
        obs = self._get_obs(data, state.info)
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_check)
        state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
        state.info["feet_air_time"] = self.update_feet_air_time(state.info)
        state.info["contact_force"] = self.update_contact_force(state)

        return state.replace(obs=obs)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_check)
        termination_check.reshape((self._num_envs, self.num_check))
        terminated = termination_check.any(axis=1)

        over_speed = np.sum(np.square(self.get_local_linvel(data)[:, :2]), axis=1) > 1e8
        terminated = terminated | over_speed
        return state.replace(
            terminated=terminated,
        )

    def update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def update_contact_force(self, state: NpEnvState):
        data = state.data
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        force = []
        for foot in self.cfg.sensor.feet:
            contact_force = self._model.get_sensor_value(foot + "_foot_contact", data)
            contact_force = quaternion.rotate_inverse(base_quat, contact_force)
            force.append(contact_force)
        return np.concatenate(force, axis=1)

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )
        return commands.astype(np.float32)

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        terminated = state.terminated

        reward_dict = self._get_reward(data, state.info)

        rewards = {k: v * self.cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        rwd = np.clip(rwd, 0.0, 10000.0)
        if "termination" in self.cfg.reward_config.scales:
            termination = self._reward_termination(terminated) * self.cfg.reward_config.scales["termination"]
            rwd += termination

        rwd = np.where(terminated, np.array(0.0), rwd)

        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        num_period = 25
        idx = generate_repeating_array(num_period, num_reset, self.period_counter)
        self.period_counter = (self.period_counter + num_reset) % num_period
        dof_pos[:, :3] = self.offset_list[idx]

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_reset, self.foot_check_num), dtype=np.bool),
            "contact_force": np.zeros((num_reset, 12), dtype=np.float32),
        }
        obs = self._get_obs(data, info)
        return obs, info

    def _get_reward(
        self,
        data: mtx.SceneData,
        info: dict,
    ) -> dict[str, np.ndarray]:
        commands = info["commands"]
        return {
            "lin_vel_z": self._reward_lin_vel_z(data),
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            "orientation": self._reward_orientation(data),
            "torques": self._reward_torques(data),
            "dof_vel": self._reward_dof_vel(data),
            "dof_acc": self._reward_dof_acc(data, info),
            "action_rate": self._reward_action_rate(info),
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "stand_still": self._reward_stand_still(data, commands),
            "hip_pos": self._reward_hip_pos(data, commands),
            "calf_pos": self._reward_calf_pos(data, commands),
            "feet_air_time": self._reward_feet_air_time(commands, info),
            "feet_stumble": self._reward_feet_stumble(data),
        }

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, data):
        # Penalize z axis base linear velocity
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1)

    def _reward_orientation(self, data):
        # Penalize non flat base orientation
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, data: mtx.SceneData):
        # Penalize torques
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        # Reward long steps
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        # reward only on first contact with the ground
        rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        # no reward for zero command
        rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
        return rew_airTime

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, data, commands: np.ndarray):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )

    def _reward_hip_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.hip_indices] - self.default_angles[self.hip_indices]),
            axis=1,
        )

    def _reward_calf_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.calf_indices] - self.default_angles[self.calf_indices]),
            axis=1,
        )

    def _reward_feet_stumble(self, data):
        # Penalize feet hitting vertical surfaces
        is_stumble = 0
        for foot in self.cfg.sensor.feet:
            contact_force = self._model.get_sensor_value(foot + "_foot_contact", data)
            is_stumble += (np.linalg.norm(contact_force, axis=1) > 5 * np.abs(contact_force[:, 2])) * 1.0
        return is_stumble
