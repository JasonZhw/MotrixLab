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
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import RM65OpenCabinetEnvCfg
from .gripper_logic import binary_hysteresis_step, raw_action_to_close_ratio


@registry.env("rm65-open-cabinet", "np")
class RM65OpenCabinetEnv(NpEnv):
    _cfg: RM65OpenCabinetEnvCfg

    def __init__(self, cfg: RM65OpenCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = -0.91
        gripper_open = self.gripper_open_pos
        gripper_mirror = -self.gripper_open_pos
        self.robot_default_joint_pos = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                gripper_open,
                gripper_mirror,
                gripper_open,
                gripper_mirror,
                gripper_mirror,
                gripper_mirror,
            ],
            np.float32,
        )
        self._gripper_mimic_multipliers = np.array([-1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        self._init_action_spaces()
        self._parse_arm_control_config()
        self._obs_noise_cfg = self._cfg.observation_noise
        self._parse_gripper_control_config()

        self._num_dof_pos = 12  # 6 arm + 6 gripper joints
        self._num_dof_vel = 12
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)
        self._init_model_handles()

        self.count = 0
        np.set_printoptions(precision=2)

    def _init_action_spaces(self) -> None:
        self._action_dim = len(self._cfg.action_scale) + 1
        self._arm_action_dim = self._action_dim - 1
        self.action_scale = np.array(self._cfg.action_scale, np.float32)
        self._action_history_len = max(int(getattr(self._cfg, "action_history_len", 0)), 0)
        base_obs_dim = self._action_dim * 2 + 7  # action_dim*2 + 7
        self._obs_dim = base_obs_dim + self._action_dim * self._action_history_len
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

    def _parse_arm_control_config(self) -> None:
        arm_cfg = self._cfg.arm_control
        ctrl_dt = float(self._cfg.ctrl_dt)

        self._arm_action_mode = arm_cfg.action_mode
        self._arm_action_in_degrees = arm_cfg.action_in_degrees
        self._arm_target_action_normalized = arm_cfg.target_action_normalized
        self._arm_use_speed_limit = arm_cfg.use_speed_limit
        self._arm_max_step = float(arm_cfg.max_joint_speed) * ctrl_dt
        self._arm_use_acc_limit = bool(arm_cfg.use_acc_limit)
        self._arm_max_acc_step = float(arm_cfg.max_joint_acc) * (ctrl_dt**2)
        self._arm_action_delay_steps = int(arm_cfg.action_delay_steps)
        self._arm_actuator_lag_alpha = float(np.clip(float(arm_cfg.actuator_lag_alpha), 0.0, 1.0))
        self._arm_delay_lag_randomization_enabled = bool(getattr(arm_cfg, "delay_lag_randomization_enabled", False))

        delay_min = int(getattr(arm_cfg, "action_delay_steps_min", self._arm_action_delay_steps))
        delay_max = int(getattr(arm_cfg, "action_delay_steps_max", self._arm_action_delay_steps))
        if delay_min > delay_max:
            delay_min, delay_max = delay_max, delay_min
        self._arm_action_delay_steps_min = max(delay_min, 0)
        self._arm_action_delay_steps_max = max(delay_max, self._arm_action_delay_steps_min)
        self._arm_action_delay_buffer_len = (
            max(int(self._arm_action_delay_steps), self._arm_action_delay_steps_max, 0) + 1
        )

        lag_min = float(getattr(arm_cfg, "actuator_lag_alpha_min", self._arm_actuator_lag_alpha))
        lag_max = float(getattr(arm_cfg, "actuator_lag_alpha_max", self._arm_actuator_lag_alpha))
        if lag_min > lag_max:
            lag_min, lag_max = lag_max, lag_min
        self._arm_actuator_lag_alpha_min = float(np.clip(lag_min, 0.0, 1.0))
        self._arm_actuator_lag_alpha_max = float(np.clip(lag_max, 0.0, 1.0))

        self._arm_speed_acc_randomization_enabled = bool(getattr(arm_cfg, "speed_acc_randomization_enabled", False))
        speed_min = float(getattr(arm_cfg, "max_joint_speed_min", arm_cfg.max_joint_speed))
        speed_max = float(getattr(arm_cfg, "max_joint_speed_max", arm_cfg.max_joint_speed))
        if speed_min > speed_max:
            speed_min, speed_max = speed_max, speed_min
        speed_min = max(speed_min, 0.0)
        speed_max = max(speed_max, speed_min)
        self._arm_max_step_min = float(speed_min * ctrl_dt)
        self._arm_max_step_max = float(speed_max * ctrl_dt)

        acc_min = float(getattr(arm_cfg, "max_joint_acc_min", arm_cfg.max_joint_acc))
        acc_max = float(getattr(arm_cfg, "max_joint_acc_max", arm_cfg.max_joint_acc))
        if acc_min > acc_max:
            acc_min, acc_max = acc_max, acc_min
        acc_min = max(acc_min, 0.0)
        acc_max = max(acc_max, acc_min)
        self._arm_max_acc_step_min = float(acc_min * (ctrl_dt**2))
        self._arm_max_acc_step_max = float(acc_max * (ctrl_dt**2))
        self._arm_target_smoothing_alpha = float(np.clip(float(arm_cfg.target_smoothing_alpha), 0.0, 1.0))

    def _parse_gripper_control_config(self) -> None:
        self._gripper_cfg = self._cfg.gripper_control
        self._gripper_action_mode = self._gripper_cfg.action_mode
        self._gripper_use_sigmoid = bool(self._gripper_cfg.use_sigmoid)
        self._gripper_close_threshold = float(self._gripper_cfg.close_threshold)
        self._gripper_close_on_threshold = float(
            getattr(self._gripper_cfg, "close_on_threshold", self._gripper_close_threshold)
        )
        self._gripper_open_off_threshold = float(
            getattr(self._gripper_cfg, "open_off_threshold", self._gripper_close_threshold)
        )
        if self._gripper_open_off_threshold > self._gripper_close_on_threshold:
            self._gripper_open_off_threshold = self._gripper_close_on_threshold
        min_switch_interval_s = float(getattr(self._gripper_cfg, "min_switch_interval_s", 0.0))
        self._gripper_min_switch_interval_steps = max(int(round(min_switch_interval_s / self._cfg.ctrl_dt)), 0)
        self._gripper_use_speed_limit = bool(self._gripper_cfg.use_speed_limit)
        self._gripper_max_step = float(self._gripper_cfg.max_speed) * self._cfg.ctrl_dt
        self._gripper_actuator_lag_alpha = float(np.clip(float(self._gripper_cfg.actuator_lag_alpha), 0.0, 1.0))

    def _init_model_handles(self) -> None:
        self.robot = self._model.get_body("base_link")
        self.gripper_tcp = self._model.get_site("gripper")
        self.left_finger_pad = self._model.get_site("left_finger_pad")
        self.right_finger_pad = self._model.get_site("right_finger_pad")
        self.robot_joint_pos_min_limit = self._model.actuator_ctrl_limits[0]
        self.robot_joint_pos_max_limit = self._model.actuator_ctrl_limits[1]
        self._obs_joint_pos_min_limit = self.robot_joint_pos_min_limit[: self._action_dim].astype(np.float32, copy=True)
        self._obs_joint_pos_max_limit = self.robot_joint_pos_max_limit[: self._action_dim].astype(np.float32, copy=True)
        self._obs_joint_pos_range = np.maximum(
            self._obs_joint_pos_max_limit - self._obs_joint_pos_min_limit,
            1e-6,
        )

        self.drawer_top_joint = self._model.get_joint("drawer_bottom_joint")
        self.drawer_top_handle = self._model.get_site("drawer_bottom_handle")

    def _compute_hold_action(self, dof_pos: np.ndarray) -> np.ndarray:
        num_envs = dof_pos.shape[0]
        arm_pos = dof_pos[:, : self._arm_action_dim]
        if self._arm_action_mode == "delta":
            arm_action = np.zeros((num_envs, self._arm_action_dim), dtype=np.float32)
        elif self._arm_action_mode == "joint_target":
            if self._arm_action_in_degrees:
                arm_action = np.rad2deg(arm_pos).astype(np.float32)
            elif self._arm_target_action_normalized:
                arm_min = self.robot_joint_pos_min_limit[: self._arm_action_dim]
                arm_max = self.robot_joint_pos_max_limit[: self._arm_action_dim]
                denom = np.maximum(arm_max - arm_min, 1e-6)
                arm_action = (2.0 * (arm_pos - arm_min) / denom - 1.0).astype(np.float32)
            else:
                arm_action = arm_pos.astype(np.float32)
        else:
            raise ValueError(f"Unsupported arm action mode: {self._arm_action_mode}")

        gripper_pos = dof_pos[:, self._arm_action_dim]
        open_pos = float(self.gripper_open_pos)
        closed_pos = float(self.gripper_closed_pos)
        gripper_range = closed_pos - open_pos
        if abs(gripper_range) < 1e-6:
            gripper_range = -1.0
        close_ratio = (gripper_pos - open_pos) / gripper_range
        close_ratio = np.clip(close_ratio, 0.0, 1.0)
        if self._gripper_action_mode == "binary":
            close_mask = close_ratio > self._gripper_close_threshold
            if self._gripper_use_sigmoid:
                gripper_action = np.where(close_mask, 10.0, -10.0)
            else:
                gripper_action = np.where(close_mask, 1.0, -1.0)
        elif self._gripper_action_mode == "continuous":
            if self._gripper_use_sigmoid:
                eps = 1e-4
                cr = np.clip(close_ratio, eps, 1.0 - eps)
                gripper_action = np.log(cr / (1.0 - cr))
            else:
                gripper_action = close_ratio * 2.0 - 1.0
        else:
            raise ValueError(f"Unsupported gripper action mode: {self._gripper_action_mode}")
        gripper_action = gripper_action.astype(np.float32)
        return np.concatenate([arm_action, gripper_action[:, None]], axis=-1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _sample_arm_delay_lag(self, num_envs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._arm_delay_lag_randomization_enabled:
            delay_steps = np.random.randint(
                self._arm_action_delay_steps_min,
                self._arm_action_delay_steps_max + 1,
                size=(num_envs,),
            ).astype(np.int32)
            lag_alpha = np.random.uniform(
                self._arm_actuator_lag_alpha_min,
                self._arm_actuator_lag_alpha_max,
                size=(num_envs,),
            ).astype(np.float32)
        else:
            delay_steps = np.full((num_envs,), int(self._arm_action_delay_steps), dtype=np.int32)
            lag_alpha = np.full((num_envs,), float(self._arm_actuator_lag_alpha), dtype=np.float32)
        if self._arm_speed_acc_randomization_enabled:
            max_step = np.random.uniform(
                self._arm_max_step_min,
                self._arm_max_step_max,
                size=(num_envs,),
            ).astype(np.float32)
            max_acc_step = np.random.uniform(
                self._arm_max_acc_step_min,
                self._arm_max_acc_step_max,
                size=(num_envs,),
            ).astype(np.float32)
        else:
            max_step = np.full((num_envs,), float(self._arm_max_step), dtype=np.float32)
            max_acc_step = np.full((num_envs,), float(self._arm_max_acc_step), dtype=np.float32)
        return delay_steps, lag_alpha, max_step, max_acc_step

    def _apply_action_delay(self, actions: np.ndarray, info: dict) -> np.ndarray:
        num_envs = actions.shape[0]
        delay_steps = info.get("arm_action_delay_steps")
        if not isinstance(delay_steps, np.ndarray) or delay_steps.shape != (num_envs,):
            delay_steps = np.full((num_envs,), int(self._arm_action_delay_steps), dtype=np.int32)
        else:
            delay_steps = delay_steps.astype(np.int32, copy=False)
        delay_steps = np.maximum(delay_steps, 0)
        buffer_len = max(int(self._arm_action_delay_buffer_len), 1)
        delay_steps = np.minimum(delay_steps, buffer_len - 1)

        buffer = info.get("action_delay_buffer")
        expected_shape = (num_envs, buffer_len, actions.shape[1])
        if buffer is None or buffer.shape != expected_shape:
            buffer = np.repeat(actions[:, None, :], buffer_len, axis=1)
        else:
            buffer = np.roll(buffer, 1, axis=1)
            buffer[:, 0, :] = actions
            steps = info.get("steps")
            if isinstance(steps, np.ndarray) and steps.shape == (num_envs,):
                reset_mask = steps == 0
                if np.any(reset_mask):
                    buffer[reset_mask] = np.repeat(actions[reset_mask][:, None, :], buffer_len, axis=1)
        info["action_delay_buffer"] = buffer
        return buffer[np.arange(num_envs), delay_steps, :]

    def _update_action_history(self, raw_actions: np.ndarray, delayed_actions: np.ndarray, info: dict) -> None:
        if self._action_history_len <= 0:
            return

        hist = info.get("action_history")
        expected_shape = (delayed_actions.shape[0], self._action_history_len, delayed_actions.shape[1])
        if hist is None or hist.shape != expected_shape:
            hist = np.repeat(raw_actions[:, None, :], self._action_history_len, axis=1)
        else:
            hist = np.roll(hist, 1, axis=1)
            hist[:, 0, :] = raw_actions
            steps = info.get("steps")
            if isinstance(steps, np.ndarray) and steps.shape == (delayed_actions.shape[0],):
                reset_mask = steps == 0
                if np.any(reset_mask):
                    hist[reset_mask] = np.repeat(raw_actions[reset_mask][:, None, :], self._action_history_len, axis=1)
        info["action_history"] = hist

    def _apply_arm_action(self, arm_action: np.ndarray, old_joint_pos: np.ndarray, info: dict) -> np.ndarray:
        arm_min_limit = self.robot_joint_pos_min_limit[: self._arm_action_dim]
        arm_max_limit = self.robot_joint_pos_max_limit[: self._arm_action_dim]
        smoothing_active = self._arm_action_mode == "joint_target" and self._arm_target_smoothing_alpha > 0.0

        if self._arm_action_mode == "delta":
            action_delta = arm_action * self.action_scale
            target_joint_pos = old_joint_pos + action_delta
            target_joint_pos = np.clip(target_joint_pos, arm_min_limit, arm_max_limit)
        elif self._arm_action_mode == "joint_target":
            if self._arm_action_in_degrees:
                target_joint_pos = np.deg2rad(arm_action)
            elif self._arm_target_action_normalized:
                arm_action = np.clip(arm_action, -1.0, 1.0)
                target_joint_pos = arm_min_limit + (arm_action + 1.0) * 0.5 * (arm_max_limit - arm_min_limit)
            else:
                target_joint_pos = arm_action
            target_joint_pos = np.clip(target_joint_pos, arm_min_limit, arm_max_limit)
            if smoothing_active:
                prev_target = info.get("arm_target_smooth", old_joint_pos)
                if not isinstance(prev_target, np.ndarray) or prev_target.shape != target_joint_pos.shape:
                    prev_target = old_joint_pos
                target_joint_pos = (
                    1.0 - self._arm_target_smoothing_alpha
                ) * prev_target + self._arm_target_smoothing_alpha * target_joint_pos
        else:
            raise ValueError(f"Unsupported arm action mode: {self._arm_action_mode}")

        action_delta = target_joint_pos - old_joint_pos
        if self._arm_use_speed_limit:
            max_step = info.get("arm_max_step")
            if isinstance(max_step, np.ndarray) and max_step.shape == (action_delta.shape[0],):
                max_step_vec = np.maximum(max_step.astype(np.float32, copy=False), 0.0)
                action_delta = np.clip(action_delta, -max_step_vec[:, None], max_step_vec[:, None])
            else:
                action_delta = np.clip(action_delta, -self._arm_max_step, self._arm_max_step)

        if self._arm_use_acc_limit:
            prev_delta = info.get("arm_prev_delta", np.zeros_like(action_delta))
            if not isinstance(prev_delta, np.ndarray) or prev_delta.shape != action_delta.shape:
                prev_delta = np.zeros_like(action_delta)
            max_delta_change = info.get("arm_max_acc_step")
            if isinstance(max_delta_change, np.ndarray) and max_delta_change.shape == (action_delta.shape[0],):
                max_delta_change_vec = np.maximum(max_delta_change.astype(np.float32, copy=False), 0.0)
                delta_change = np.clip(
                    action_delta - prev_delta,
                    -max_delta_change_vec[:, None],
                    max_delta_change_vec[:, None],
                )
                action_delta = prev_delta + delta_change
            else:
                max_delta_change_scalar = float(self._arm_max_acc_step)
                if max_delta_change_scalar > 0.0:
                    delta_change = np.clip(
                        action_delta - prev_delta,
                        -max_delta_change_scalar,
                        max_delta_change_scalar,
                    )
                    action_delta = prev_delta + delta_change
                else:
                    action_delta = prev_delta

        info["arm_prev_delta"] = action_delta
        target_joint_pos = old_joint_pos + action_delta
        if smoothing_active:
            info["arm_target_smooth"] = target_joint_pos

        lag_alpha = info.get("arm_actuator_lag_alpha", self._arm_actuator_lag_alpha)
        if isinstance(lag_alpha, np.ndarray) and lag_alpha.shape == (target_joint_pos.shape[0],):
            lag_alpha_vec = np.clip(lag_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        else:
            lag_alpha_vec = np.full((target_joint_pos.shape[0],), float(self._arm_actuator_lag_alpha), dtype=np.float32)
        if np.any(lag_alpha_vec > 0.0):
            prev_cmd = info.get("arm_actuator_target", old_joint_pos)
            if not isinstance(prev_cmd, np.ndarray) or prev_cmd.shape != target_joint_pos.shape:
                prev_cmd = old_joint_pos
            target_joint_pos = (1.0 - lag_alpha_vec[:, None]) * prev_cmd + (lag_alpha_vec[:, None] * target_joint_pos)

        target_joint_pos = target_joint_pos.astype(np.float32, copy=False)
        info["arm_actuator_target"] = target_joint_pos
        return target_joint_pos

    def _apply_gripper_action(self, gripper_action: np.ndarray, info: dict) -> np.ndarray:
        close_ratio = raw_action_to_close_ratio(gripper_action, use_sigmoid=self._gripper_use_sigmoid)
        gripper_closed_cmd = None
        if self._gripper_action_mode == "binary":
            prev_closed = info.get("gripper_binary_closed")
            if not isinstance(prev_closed, np.ndarray) or prev_closed.shape != close_ratio.shape:
                prev_closed = close_ratio > self._gripper_close_on_threshold
            steps_since_switch = info.get("gripper_steps_since_switch")
            if not isinstance(steps_since_switch, np.ndarray) or steps_since_switch.shape != close_ratio.shape:
                steps_since_switch = np.full(
                    close_ratio.shape,
                    self._gripper_min_switch_interval_steps,
                    dtype=np.int32,
                )
            gripper_closed_cmd, switched = binary_hysteresis_step(
                close_ratio=close_ratio,
                prev_closed=prev_closed,
                steps_since_switch=steps_since_switch,
                close_on_threshold=self._gripper_close_on_threshold,
                open_off_threshold=self._gripper_open_off_threshold,
                min_switch_interval_steps=self._gripper_min_switch_interval_steps,
            )
            steps_since_switch = np.where(switched, 0, steps_since_switch + 1).astype(np.int32)
            info["gripper_binary_closed"] = gripper_closed_cmd
            info["gripper_steps_since_switch"] = steps_since_switch
            gripper_pos = np.where(gripper_closed_cmd, self.gripper_closed_pos, self.gripper_open_pos)
        elif self._gripper_action_mode == "continuous":
            gripper_pos = self.gripper_open_pos + (self.gripper_closed_pos - self.gripper_open_pos) * close_ratio
            gripper_closed_cmd = close_ratio > self._gripper_close_threshold
        else:
            raise ValueError(f"Unsupported gripper action mode: {self._gripper_action_mode}")

        prev_gripper = info.get("gripper_target_smooth", gripper_pos)
        if not isinstance(prev_gripper, np.ndarray) or prev_gripper.shape != gripper_pos.shape:
            prev_gripper = gripper_pos
        if self._gripper_use_speed_limit and self._gripper_max_step > 0.0:
            delta = np.clip(gripper_pos - prev_gripper, -self._gripper_max_step, self._gripper_max_step)
            gripper_pos = prev_gripper + delta
        if self._gripper_actuator_lag_alpha > 0.0:
            gripper_pos = (1.0 - self._gripper_actuator_lag_alpha) * prev_gripper + (
                self._gripper_actuator_lag_alpha * gripper_pos
            )

        gripper_pos = gripper_pos.astype(np.float32, copy=False)
        info["gripper_target_smooth"] = gripper_pos
        if gripper_closed_cmd is not None:
            info["gripper_closed_cmd"] = np.asarray(gripper_closed_cmd, dtype=bool)
        info["gripper_close_ratio"] = np.asarray(close_ratio, dtype=np.float32)
        info["current_gripper_action"] = gripper_pos
        return gripper_pos[:, None]

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        assert not np.isnan(actions).any(), "actions contain nan"

        raw_actions = np.array(actions, copy=True)
        delayed_actions = self._apply_action_delay(actions, state.info)
        self._update_action_history(raw_actions, delayed_actions, state.info)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = delayed_actions

        old_joint_pos = self.get_robot_joint_pos(state.data)[:, : self._arm_action_dim]
        target_joint_pos = self._apply_arm_action(delayed_actions[:, : self._arm_action_dim], old_joint_pos, state.info)
        gripper_action_cmd = self._apply_gripper_action(delayed_actions[:, -1], state.info)

        new_pos = np.concatenate([target_joint_pos, gripper_action_cmd], axis=-1)

        # step action
        cliped_new_pos = np.clip(
            new_pos, self.robot_joint_pos_min_limit, self.robot_joint_pos_max_limit, dtype=np.float32
        )  # clip new pos to limit

        # actuator1~8 by order
        state.data.actuator_ctrls = cliped_new_pos

        return state

    def update_state(self, state: NpEnvState):
        self._enforce_drawer_grasp_constraint(state)
        # compute obs
        obs = self._compute_observation(state.data, state.info)
        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.obs = obs
        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state

    def _enforce_drawer_grasp_constraint(self, state: NpEnvState):
        reward_cfg = self._cfg.reward
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)
        gripper_drawer_dist = np.linalg.norm(drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1)
        gripper_range = max(abs(self.gripper_open_pos - self.gripper_closed_pos), 1e-6)
        close_ratio = np.clip(
            (self.gripper_open_pos - state.info["current_gripper_action"]) / gripper_range,
            0.0,
            1.0,
        )
        left_z = self.left_finger_pad.get_pose(state.data)[:, 2]
        right_z = self.right_finger_pad.get_pose(state.data)[:, 2]
        align_mask = np.logical_and(drawer_grasp_pose[:, 2] - left_z >= 0.0, right_z - drawer_grasp_pose[:, 2] >= 0.0)
        grasp_candidate = np.logical_and(
            gripper_drawer_dist < reward_cfg.grasp_dist,
            close_ratio > reward_cfg.grasp_close_ratio,
        )
        grasp_candidate = np.logical_and(grasp_candidate, align_mask)

        hold_steps = state.info.get("grasp_hold_steps")
        if not isinstance(hold_steps, np.ndarray) or hold_steps.shape != grasp_candidate.shape:
            hold_steps = np.zeros_like(grasp_candidate, dtype=np.int32)
        hold_steps = np.where(grasp_candidate, hold_steps + 1, 0)
        required_steps = max(int(getattr(reward_cfg, "grasp_hold_steps", 1)), 1)
        grasped = hold_steps >= required_steps

        state.info["grasp_hold_steps"] = hold_steps
        state.info["grasped"] = grasped
        phase2_mask = state.info.get("phase2_mask", grasped)
        phase2_mask = np.logical_or(phase2_mask, grasped)
        state.info["phase2_mask"] = phase2_mask

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]

        noise_scale = self._cfg.reset.joint_pos_noise_scale
        noise_pos = np.random.uniform(-noise_scale, noise_scale, (num_reset, self._action_dim))

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_pos[:, : self._action_dim] += noise_pos  # Add noise in range [-0.125, 0.125]
        gripper_left = dof_pos[:, self._action_dim - 1]
        mimic_pos = gripper_left[:, None] * self._gripper_mimic_multipliers[None, :]
        dof_pos[:, self._action_dim :] = mimic_pos
        data.reset(self._model)
        data.set_dof_vel(np.zeros((num_reset, self._num_dof_vel + 4), dtype=np.float32))  # Includes robot and cabinet
        data.set_dof_pos(np.concatenate([dof_pos, np.zeros((num_reset, 4), dtype=np.float32)], axis=-1), self._model)
        self._model.forward_kinematic(data)

        hold_action = self._compute_hold_action(dof_pos)
        gripper_range = max(abs(self.gripper_open_pos - self.gripper_closed_pos), 1e-6)
        init_close_ratio = np.clip((self.gripper_open_pos - gripper_left) / gripper_range, 0.0, 1.0).astype(np.float32)
        init_binary_closed = init_close_ratio > self._gripper_close_on_threshold
        arm_delay_steps, arm_lag_alpha, arm_max_step, arm_max_acc_step = self._sample_arm_delay_lag(num_reset)
        obs_noise_cfg = self._obs_noise_cfg
        bias_pos = np.zeros((num_reset, 3), dtype=np.float32)
        if obs_noise_cfg.target_pos_bias_std > 0.0:
            bias_pos = np.random.normal(0.0, obs_noise_cfg.target_pos_bias_std, size=(num_reset, 3)).astype(np.float32)
        bias_quat = self._sample_quat_bias(num_reset, obs_noise_cfg.target_rot_bias_std)
        handle_pose = self.drawer_top_handle.get_pose(data).astype(np.float32)
        info = {
            "current_actions": hold_action.copy(),
            "last_actions": hold_action.copy(),
            "phase2_mask": np.zeros(num_reset, dtype=bool),  # 1D array
            "grasped": np.zeros(num_reset, dtype=bool),
            "grasp_hold_steps": np.zeros(num_reset, dtype=np.int32),
            "current_gripper_action": np.full(num_reset, self.gripper_open_pos, dtype=np.float32),  # 1D array
            "obs_handle_bias_pos": bias_pos,
            "obs_handle_bias_quat": bias_quat,
            "obs_handle_pose_last": handle_pose.copy(),
            "handle_pose_override": np.zeros((num_reset, handle_pose.shape[1]), dtype=np.float32),
            "handle_pose_override_mask": np.zeros(num_reset, dtype=bool),
            "arm_action_delay_steps": arm_delay_steps,
            "arm_actuator_lag_alpha": arm_lag_alpha,
            "arm_max_step": arm_max_step,
            "arm_max_acc_step": arm_max_acc_step,
            "arm_target_smooth": dof_pos[:, : self._arm_action_dim],
            "arm_prev_delta": np.zeros((num_reset, self._arm_action_dim), dtype=np.float32),
            "arm_actuator_target": dof_pos[:, : self._arm_action_dim],
            "gripper_target_smooth": gripper_left.copy(),
            "gripper_binary_closed": init_binary_closed.astype(bool, copy=False),
            "gripper_closed_cmd": init_binary_closed.astype(bool, copy=False),
            "gripper_steps_since_switch": np.full(
                num_reset,
                self._gripper_min_switch_interval_steps,
                dtype=np.int32,
            ),
            "gripper_close_ratio": init_close_ratio.copy(),
            "prev_gripper_closed_cmd": init_binary_closed.astype(bool, copy=True),
            "prev_open_dist": np.zeros(num_reset, dtype=np.float32),
            "open_bonus_progress": np.zeros(num_reset, dtype=np.int32),
            "Reward": {
                "dist": np.zeros(num_reset, dtype=np.float32),
                "quat": np.zeros(num_reset, dtype=np.float32),
                "close_gripper": np.zeros(num_reset, dtype=np.float32),
                "open_reward": np.zeros(num_reset, dtype=np.float32),
                "open_delta_reward": np.zeros(num_reset, dtype=np.float32),
                "slip_penalty": np.zeros(num_reset, dtype=np.float32),
                "finger_penalty": np.zeros(num_reset, dtype=np.float32),
                "action_penalty": np.zeros(num_reset, dtype=np.float32),
                "joint_vel_penalty": np.zeros(num_reset, dtype=np.float32),
                "gripper_switch_penalty": np.zeros(num_reset, dtype=np.float32),
                "truncation_penalty": np.zeros(num_reset, dtype=np.float32),
            },
            "metrics": {
                "open_dist": np.zeros(num_reset, dtype=np.float32),
                "gripper_drawer_dist": np.zeros(num_reset, dtype=np.float32),
                "gripper_close_rate": np.zeros(num_reset, dtype=np.float32),
                "close_amount": np.zeros(num_reset, dtype=np.float32),
                "wrong_open": np.zeros(num_reset, dtype=np.float32),
                "gripper_switch": np.zeros(num_reset, dtype=np.float32),
                "action_penalty_rate": np.zeros(num_reset, dtype=np.float32),
                "joint_vel_penalty_rate": np.zeros(num_reset, dtype=np.float32),
            },
        }
        buffer_len = max(int(self._arm_action_delay_buffer_len), 1)
        info["action_delay_buffer"] = np.repeat(hold_action[:, None, :], buffer_len, axis=1)
        if self._action_history_len > 0:
            info["action_history"] = np.repeat(hold_action[:, None, :], self._action_history_len, axis=1)
        latency_steps = max(int(obs_noise_cfg.latency_steps), 0)
        if latency_steps > 0:
            info["obs_handle_pose_buffer"] = np.repeat(handle_pose[:, None, :], latency_steps + 1, axis=1)
        obs = self._compute_observation(data, info)
        return obs, info

    def _compute_observation(self, data: mtx.SceneData, info: dict):
        num_envs = data.shape[0]
        obs_noise_cfg = self._obs_noise_cfg

        # dof_pos: (num_envs, 7) range: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(data)  # shape: (num_envs, 7)
        dof_pos_rel_raw = self._get_robot_joint_pos_rel(dof_pos)[:, : self._action_dim]
        dof_pos_rel = dof_pos_rel_raw.copy()
        if obs_noise_cfg.enabled and obs_noise_cfg.joint_noise_enabled and obs_noise_cfg.joint_pos_std > 0.0:
            dof_pos_rel = dof_pos_rel + np.random.normal(
                0.0, obs_noise_cfg.joint_pos_std, size=dof_pos_rel.shape
            ).astype(np.float32)

        dof_pos_abs = dof_pos_rel + self.robot_default_joint_pos[: self._action_dim]
        dof_pos_abs_raw = dof_pos_rel_raw + self.robot_default_joint_pos[: self._action_dim]
        dof_pos_scaled = 2.0 * (dof_pos_abs - self._obs_joint_pos_min_limit) / self._obs_joint_pos_range - 1.0
        # relative vel: finite-difference from consecutive joint positions
        # to reduce dependency on simulator/driver-specific velocity channels.
        dt = max(float(self._cfg.ctrl_dt), 1e-6)
        prev_dof_pos_abs_raw = info.get("obs_prev_dof_pos_abs_raw")
        if isinstance(prev_dof_pos_abs_raw, np.ndarray) and prev_dof_pos_abs_raw.shape == dof_pos_abs_raw.shape:
            dof_vel_rel = (dof_pos_abs_raw - prev_dof_pos_abs_raw) / dt
        else:
            dof_vel_rel = np.zeros_like(dof_pos_abs_raw, dtype=np.float32)

        steps = info.get("steps")
        if isinstance(steps, np.ndarray) and steps.shape == (num_envs,):
            reset_mask = steps == 0
            if np.any(reset_mask):
                dof_vel_rel[reset_mask] = 0.0
        info["obs_prev_dof_pos_abs_raw"] = dof_pos_abs_raw.astype(np.float32, copy=True)

        if obs_noise_cfg.enabled and obs_noise_cfg.joint_noise_enabled and obs_noise_cfg.joint_vel_std > 0.0:
            dof_vel_rel = dof_vel_rel + np.random.normal(
                0.0, obs_noise_cfg.joint_vel_std, size=dof_vel_rel.shape
            ).astype(np.float32)
        dof_vel_rel = dof_vel_rel / 2

        # relative pose: position delta + relative quaternion (target * current.inverse)
        robot_grasp_pose = self.gripper_tcp.get_pose(data)
        drawer_grasp_pose = self._resolve_handle_pose(data, info)
        if obs_noise_cfg.enabled and obs_noise_cfg.handle_pose_noise_enabled:
            drawer_grasp_pose = self._get_noisy_handle_pose(drawer_grasp_pose, info)
        pos_delta = drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3]
        quat_target = drawer_grasp_pose[:, 3:]
        quat_current = robot_grasp_pose[:, 3:]
        q_rel = quaternion.mul(quat_target, quaternion.inverse(quat_current))
        q_norm = np.linalg.norm(q_rel, axis=-1, keepdims=True)
        q_rel = q_rel / np.maximum(q_norm, 1e-6)
        # Enforce a consistent hemisphere to avoid sign flips.
        sign = np.where(q_rel[:, 3:4] < 0.0, -1.0, 1.0)
        q_rel = q_rel * sign
        to_target = np.concatenate([pos_delta, q_rel], axis=-1)

        obs = np.concatenate([dof_pos_scaled, dof_vel_rel, to_target], axis=-1)
        if self._action_history_len > 0:
            history = info.get("action_history")
            expected_shape = (num_envs, self._action_history_len, self._action_dim)
            if history is None or history.shape != expected_shape:
                history = np.zeros(expected_shape, dtype=np.float32)
            else:
                history = history.astype(np.float32, copy=False)
            obs = np.concatenate([obs, history.reshape(num_envs, -1)], axis=-1)

        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        return np.clip(obs, -5, 5)

    def _resolve_handle_pose(self, data: mtx.SceneData, info: dict):
        handle_pose = self.drawer_top_handle.get_pose(data)
        override = info.get("handle_pose_override")
        if override is not None:
            override_pose = np.asarray(override, dtype=np.float32)
            if override_pose.ndim == 1 and override_pose.shape[0] == handle_pose.shape[1]:
                override_pose = np.tile(override_pose, (handle_pose.shape[0], 1))
            if override_pose.shape != handle_pose.shape:
                return handle_pose
            override_mask = info.get("handle_pose_override_mask")
            if override_mask is not None:
                mask = np.asarray(override_mask, dtype=bool)
                if mask.shape != (handle_pose.shape[0],):
                    return handle_pose
                if np.any(mask):
                    return np.where(mask[:, None], override_pose, handle_pose)
                if np.any(override_pose):
                    return override_pose
                return handle_pose
            return override_pose
        return handle_pose

    def _sample_quat_bias(self, num_envs: int, rot_std: float):
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        base = np.tile(identity, (num_envs, 1))
        if rot_std <= 0.0:
            return base
        return self._apply_quat_noise(base, rot_std)

    def _apply_quat_noise(self, quat: np.ndarray, rot_std: float):
        num_envs = quat.shape[0]
        axes = np.random.normal(0.0, 1.0, (num_envs, 3)).astype(np.float32)
        axis_norm = np.linalg.norm(axes, axis=-1, keepdims=True)
        safe_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axes = np.where(axis_norm < 1e-6, safe_axis, axes)
        axes = axes / np.maximum(axis_norm, 1e-6)
        angles = np.random.normal(0.0, rot_std, (num_envs,)).astype(np.float32)
        half_angles = angles * 0.5
        sin_half = np.sin(half_angles).astype(np.float32)
        cos_half = np.cos(half_angles).astype(np.float32)
        delta_q = np.concatenate([axes * sin_half[:, None], cos_half[:, None]], axis=-1)
        noisy_quat = quaternion.mul(delta_q, quat)
        quat_norm = np.linalg.norm(noisy_quat, axis=-1, keepdims=True)
        return noisy_quat / np.maximum(quat_norm, 1e-6)

    def _get_noisy_handle_pose(self, handle_pose: np.ndarray, info: dict):
        cfg = self._obs_noise_cfg
        if not cfg.enabled or not cfg.handle_pose_noise_enabled:
            return handle_pose

        num_envs = handle_pose.shape[0]
        noisy_pose = handle_pose.copy()

        bias_resample_prob = float(np.clip(cfg.bias_resample_prob, 0.0, 1.0))
        if cfg.target_pos_bias_std > 0.0:
            bias_pos = info.get("obs_handle_bias_pos")
            if not isinstance(bias_pos, np.ndarray) or bias_pos.shape != (num_envs, 3):
                bias_pos = np.random.normal(0.0, cfg.target_pos_bias_std, size=(num_envs, 3)).astype(np.float32)
            if bias_resample_prob > 0.0:
                resample_mask = np.random.rand(num_envs) < bias_resample_prob
                if np.any(resample_mask):
                    bias_pos[resample_mask] = np.random.normal(
                        0.0, cfg.target_pos_bias_std, size=(resample_mask.sum(), 3)
                    ).astype(np.float32)
            info["obs_handle_bias_pos"] = bias_pos
            noisy_pose[:, :3] = noisy_pose[:, :3] + bias_pos

        if cfg.target_rot_bias_std > 0.0:
            bias_quat = info.get("obs_handle_bias_quat")
            if not isinstance(bias_quat, np.ndarray) or bias_quat.shape != (num_envs, 4):
                bias_quat = self._sample_quat_bias(num_envs, cfg.target_rot_bias_std)
            if bias_resample_prob > 0.0:
                resample_mask = np.random.rand(num_envs) < bias_resample_prob
                if np.any(resample_mask):
                    bias_quat[resample_mask] = self._sample_quat_bias(int(resample_mask.sum()), cfg.target_rot_bias_std)
            info["obs_handle_bias_quat"] = bias_quat
            noisy_pose[:, 3:] = quaternion.mul(bias_quat, noisy_pose[:, 3:])

        if cfg.target_pos_std > 0.0:
            noisy_pose[:, :3] = noisy_pose[:, :3] + np.random.normal(
                0.0, cfg.target_pos_std, size=(num_envs, 3)
            ).astype(np.float32)
        if cfg.target_rot_std > 0.0:
            noisy_pose[:, 3:] = self._apply_quat_noise(noisy_pose[:, 3:], cfg.target_rot_std)

        dropout_prob = float(np.clip(cfg.dropout_prob, 0.0, 1.0))
        if dropout_prob > 0.0:
            dropout_mask = np.random.rand(num_envs) < dropout_prob
            if np.any(dropout_mask):
                last_pose = info.get("obs_handle_pose_last")
                if cfg.hold_last_on_dropout and isinstance(last_pose, np.ndarray):
                    noisy_pose[dropout_mask] = last_pose[dropout_mask]
                else:
                    noisy_pose[dropout_mask] = handle_pose[dropout_mask]

        info["obs_handle_pose_last"] = noisy_pose.copy()

        latency_steps = max(int(cfg.latency_steps), 0)
        if latency_steps > 0:
            buffer = info.get("obs_handle_pose_buffer")
            expected_shape = (num_envs, latency_steps + 1, noisy_pose.shape[1])
            if buffer is None or buffer.shape != expected_shape:
                buffer = np.repeat(noisy_pose[:, None, :], latency_steps + 1, axis=1)
            else:
                buffer = np.roll(buffer, 1, axis=1)
                buffer[:, 0, :] = noisy_pose
            info["obs_handle_pose_buffer"] = buffer
            return buffer[:, -1, :]

        return noisy_pose

    def _compute_distance_alignment_terms(
        self,
        state: NpEnvState,
        reward_cfg,
        robot_grasp_pose: np.ndarray,
        drawer_grasp_pose: np.ndarray,
        gripper_drawer_dist: np.ndarray,
    ) -> dict[str, np.ndarray]:
        dist_reward = 1 - np.tanh(gripper_drawer_dist / reward_cfg.dist_std)
        dist_reward *= reward_cfg.dist_scale

        quat_reward = quaternion.similarity(robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:])
        if reward_cfg.quat_reward_dist_thresh > 0.0:
            quat_reward = np.where(gripper_drawer_dist < reward_cfg.quat_reward_dist_thresh, quat_reward, 0.0)
        quat_reward = quat_reward * reward_cfg.quat_reward_scale

        lfinger_dist = drawer_grasp_pose[:, 2] - self.left_finger_pad.get_pose(state.data)[:, 2]
        rfinger_dist = self.right_finger_pad.get_pose(state.data)[:, 2] - drawer_grasp_pose[:, 2]
        align_mask = np.logical_and(lfinger_dist >= 0.0, rfinger_dist >= 0.0)

        gripper_range = max(abs(self.gripper_open_pos - self.gripper_closed_pos), 1e-6)
        close_amount_raw = self.gripper_open_pos - state.info["current_gripper_action"]
        close_amount_raw = np.clip(close_amount_raw, 0.0, gripper_range)
        close_ratio = close_amount_raw / gripper_range
        close_amount = close_amount_raw * (0.04 / gripper_range)
        close_gripper = (
            np.where(
                np.logical_and(gripper_drawer_dist < reward_cfg.gripper_close_dist, align_mask),
                reward_cfg.gripper_close_reward,
                reward_cfg.gripper_close_penalty,
            )
            * close_amount
        )

        return {
            "dist_reward": dist_reward,
            "quat_reward": quat_reward,
            "lfinger_dist": lfinger_dist,
            "rfinger_dist": rfinger_dist,
            "align_mask": align_mask,
            "close_ratio": close_ratio,
            "close_amount": close_amount,
            "close_gripper": close_gripper,
        }

    def _compute_open_reward_terms(
        self,
        state: NpEnvState,
        reward_cfg,
        gripper_drawer_dist: np.ndarray,
        align_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        open_dist = self.drawer_top_joint.get_dof_pos(state.data)
        open_dist = np.asarray(open_dist).reshape(-1)
        open_dist = np.clip(open_dist, 0.0, 1.0)

        open_reward = (np.exp(open_dist) - 1.0) * reward_cfg.open_reward_scale
        if reward_cfg.wrong_open_dist > 0.0:
            wrong_open = np.logical_and(open_dist > 0.0, gripper_drawer_dist > reward_cfg.wrong_open_dist)
        else:
            wrong_open = np.zeros_like(open_dist, dtype=bool)
        open_reward = np.where(np.logical_not(wrong_open), open_reward, 0.0)

        grasped = state.info.get("grasped")
        if grasped is None:
            grasped = align_mask
        phase2_mask = state.info.get("phase2_mask")
        if not isinstance(phase2_mask, np.ndarray) or phase2_mask.shape != grasped.shape:
            phase2_mask = grasped

        strict_open_gate = np.logical_or(grasped, phase2_mask)
        strict_open_dist = float(getattr(reward_cfg, "open_reward_strict_dist", 0.0))
        if strict_open_dist > 0.0:
            near_mask = gripper_drawer_dist < strict_open_dist
        else:
            near_mask = np.ones_like(open_dist, dtype=bool)

        open_gate = np.logical_and(strict_open_gate, near_mask)
        open_reward = np.where(open_gate, open_reward, 0.0)

        prev_open_dist = state.info.get("prev_open_dist")
        if not isinstance(prev_open_dist, np.ndarray) or prev_open_dist.shape != open_dist.shape:
            prev_open_dist = np.zeros_like(open_dist, dtype=np.float32)
        open_delta = np.clip(open_dist - prev_open_dist, 0.0, None)
        open_delta_reward = open_delta * reward_cfg.open_delta_reward_scale
        open_delta_reward = np.where(open_gate, open_delta_reward, 0.0)
        open_delta_reward = np.where(np.logical_not(wrong_open), open_delta_reward, 0.0)

        state.info["prev_open_dist"] = open_dist.astype(np.float32, copy=True)

        return {
            "open_dist": open_dist,
            "wrong_open": wrong_open,
            "grasped": grasped,
            "phase2_mask": phase2_mask,
            "open_reward": open_reward,
            "open_delta_reward": open_delta_reward,
        }

    def _compute_progress_reward_terms(
        self,
        state: NpEnvState,
        reward_cfg,
        open_dist: np.ndarray,
        grasped: np.ndarray,
        phase2_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        grasp_hold_reward_scale = float(getattr(reward_cfg, "grasp_hold_reward_scale", 0.0))
        grasp_hold_open_scale = float(getattr(reward_cfg, "grasp_hold_open_scale", 0.0))
        grasp_hold_reward = np.where(grasped, grasp_hold_reward_scale + grasp_hold_open_scale * open_dist, 0.0)

        prev_open_bonus = state.info.get("open_bonus_progress")
        if not isinstance(prev_open_bonus, np.ndarray) or prev_open_bonus.shape != open_dist.shape:
            prev_open_bonus = np.zeros_like(open_dist, dtype=np.int32)

        bonus1_dist = float(getattr(reward_cfg, "open_bonus_dist_1", 0.0))
        bonus1_reward = float(getattr(reward_cfg, "open_bonus_reward_1", 0.0))
        bonus2_dist = float(getattr(reward_cfg, "open_bonus_dist_2", 0.0))
        bonus2_reward = float(getattr(reward_cfg, "open_bonus_reward_2", 0.0))

        bonus_progress = prev_open_bonus.copy()
        pass_bonus1 = np.logical_and(open_dist >= bonus1_dist, bonus_progress < 1)
        pass_bonus2 = np.logical_and(open_dist >= bonus2_dist, bonus_progress < 2)
        open_bonus_reward = np.where(pass_bonus1, bonus1_reward, 0.0)
        open_bonus_reward += np.where(pass_bonus2, bonus2_reward, 0.0)
        open_bonus_reward = np.where(grasped, open_bonus_reward, 0.0)
        bonus_progress = np.where(pass_bonus1, 1, bonus_progress)
        bonus_progress = np.where(pass_bonus2, 2, bonus_progress)
        state.info["open_bonus_progress"] = bonus_progress.astype(np.int32)

        slip_open_dist_thresh = float(np.clip(reward_cfg.slip_open_dist_thresh, 0.0, 1.0))
        slipped = np.logical_and(
            np.logical_and(phase2_mask, np.logical_not(grasped)),
            open_dist > slip_open_dist_thresh,
        )
        slip_penalty_open_scale = float(getattr(reward_cfg, "slip_penalty_open_scale", 0.0))
        slip_penalty = np.where(
            slipped,
            -(reward_cfg.slip_penalty + slip_penalty_open_scale * open_dist),
            0.0,
        )

        return {
            "grasp_hold_reward": grasp_hold_reward,
            "open_bonus_reward": open_bonus_reward,
            "slip_penalty": slip_penalty,
        }

    def _compute_penalty_terms(
        self,
        state: NpEnvState,
        reward_cfg,
        gripper_drawer_dist: np.ndarray,
        lfinger_dist: np.ndarray,
        rfinger_dist: np.ndarray,
        align_mask: np.ndarray,
        close_amount: np.ndarray,
        close_ratio: np.ndarray,
        open_dist: np.ndarray,
    ) -> dict[str, np.ndarray]:
        action_penalty = np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=-1)
        joint_vel_penalty = np.sum(np.square(state.data.dof_vel[:, : self._action_dim]), axis=-1)

        finger_penalty = np.zeros_like(lfinger_dist)
        finger_penalty += np.where(lfinger_dist < 0.0, lfinger_dist, 0.0)
        finger_penalty += np.where(rfinger_dist < 0.0, rfinger_dist, 0.0)
        finger_penalty = finger_penalty * reward_cfg.finger_penalty_weight
        close_mask = gripper_drawer_dist < reward_cfg.finger_penalty_dist
        finger_penalty = np.where(close_mask, finger_penalty, 0.0)

        align_open_mask = close_amount < reward_cfg.finger_align_close_amount_thresh
        finger_align_reward = np.where(
            np.logical_and(close_mask, np.logical_and(align_mask, align_open_mask)),
            reward_cfg.finger_align_reward,
            0.0,
        )

        gripper_closed_cmd = state.info.get("gripper_closed_cmd")
        if not isinstance(gripper_closed_cmd, np.ndarray) or gripper_closed_cmd.shape != close_ratio.shape:
            gripper_closed_cmd = close_ratio > self._gripper_close_threshold
        gripper_closed_cmd = np.asarray(gripper_closed_cmd, dtype=bool)
        prev_gripper_closed_cmd = state.info.get("prev_gripper_closed_cmd")
        if not isinstance(prev_gripper_closed_cmd, np.ndarray) or (
            prev_gripper_closed_cmd.shape != gripper_closed_cmd.shape
        ):
            prev_gripper_closed_cmd = gripper_closed_cmd.copy()
        switch_mask = gripper_closed_cmd != prev_gripper_closed_cmd
        switch_penalty_dist = float(getattr(reward_cfg, "gripper_switch_penalty_dist", 0.0))
        if switch_penalty_dist > 0.0:
            switch_gate = gripper_drawer_dist < switch_penalty_dist
        else:
            switch_gate = np.ones_like(switch_mask, dtype=bool)
        gripper_switch_penalty_scale = float(getattr(reward_cfg, "gripper_switch_penalty", 0.0))
        gripper_switch_penalty = np.where(
            np.logical_and(switch_mask, switch_gate),
            -gripper_switch_penalty_scale,
            0.0,
        ).astype(np.float32)
        state.info["prev_gripper_closed_cmd"] = gripper_closed_cmd.copy()

        if self.count < reward_cfg.action_penalty_switch_step:
            action_penalty_rate = reward_cfg.action_penalty_rate_early
            joint_vel_penalty_rate = reward_cfg.joint_vel_penalty_rate_early
        else:
            action_penalty_rate = reward_cfg.action_penalty_rate_late
            joint_vel_penalty_rate = reward_cfg.joint_vel_penalty_rate_late

        action_penalty_term = -action_penalty_rate * action_penalty
        joint_vel_penalty_term = -joint_vel_penalty_rate * joint_vel_penalty

        return {
            "finger_penalty": finger_penalty,
            "finger_align_reward": finger_align_reward,
            "gripper_switch_penalty": gripper_switch_penalty,
            "switch_mask": switch_mask,
            "action_penalty_term": action_penalty_term,
            "joint_vel_penalty_term": joint_vel_penalty_term,
            "action_penalty_rate": np.full_like(open_dist, action_penalty_rate, dtype=np.float32),
            "joint_vel_penalty_rate": np.full_like(open_dist, joint_vel_penalty_rate, dtype=np.float32),
        }

    def _update_reward_info(
        self,
        state: NpEnvState,
        *,
        reward_cfg,
        alignment_terms: dict[str, np.ndarray],
        open_terms: dict[str, np.ndarray],
        progress_terms: dict[str, np.ndarray],
        penalty_terms: dict[str, np.ndarray],
        gripper_drawer_dist: np.ndarray,
        truncation_penalty: np.ndarray,
    ) -> None:
        grasped = open_terms["grasped"]
        state.info["Reward"] = {
            "dist": alignment_terms["dist_reward"],
            "quat": alignment_terms["quat_reward"],
            "close_gripper": alignment_terms["close_gripper"],
            "open_reward": open_terms["open_reward"],
            "open_delta_reward": open_terms["open_delta_reward"],
            "grasp_hold_reward": progress_terms["grasp_hold_reward"],
            "open_bonus_reward": progress_terms["open_bonus_reward"],
            "slip_penalty": progress_terms["slip_penalty"],
            "finger_penalty": penalty_terms["finger_penalty"],
            "finger_align_reward": penalty_terms["finger_align_reward"],
            "gripper_switch_penalty": penalty_terms["gripper_switch_penalty"],
            "grasped_rate": grasped.astype(np.float32),
            "action_penalty": penalty_terms["action_penalty_term"],
            "joint_vel_penalty": penalty_terms["joint_vel_penalty_term"],
            "truncation_penalty": truncation_penalty,
        }
        state.info["metrics"] = {
            "open_dist": open_terms["open_dist"],
            "gripper_drawer_dist": gripper_drawer_dist,
            "gripper_close_rate": (gripper_drawer_dist < reward_cfg.gripper_close_dist).astype(np.float32),
            "grasp_dist_hit_rate": (gripper_drawer_dist < reward_cfg.grasp_dist).astype(np.float32),
            "grasp_close_hit_rate": (alignment_terms["close_ratio"] > reward_cfg.grasp_close_ratio).astype(np.float32),
            "grasp_align_hit_rate": alignment_terms["align_mask"].astype(np.float32),
            "close_amount": alignment_terms["close_amount"],
            "wrong_open": open_terms["wrong_open"].astype(np.float32),
            "gripper_switch": penalty_terms["switch_mask"].astype(np.float32),
            "grasped_rate": grasped.astype(np.float32),
            "action_penalty_rate": penalty_terms["action_penalty_rate"],
            "joint_vel_penalty_rate": penalty_terms["joint_vel_penalty_rate"],
        }

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray):
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)

        gripper_drawer_dist = np.linalg.norm(drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1)
        reward_cfg = self._cfg.reward
        alignment_terms = self._compute_distance_alignment_terms(
            state,
            reward_cfg,
            robot_grasp_pose,
            drawer_grasp_pose,
            gripper_drawer_dist,
        )
        open_terms = self._compute_open_reward_terms(
            state,
            reward_cfg,
            gripper_drawer_dist,
            alignment_terms["align_mask"],
        )
        progress_terms = self._compute_progress_reward_terms(
            state,
            reward_cfg,
            open_terms["open_dist"],
            open_terms["grasped"],
            open_terms["phase2_mask"],
        )
        penalty_terms = self._compute_penalty_terms(
            state,
            reward_cfg,
            gripper_drawer_dist,
            alignment_terms["lfinger_dist"],
            alignment_terms["rfinger_dist"],
            alignment_terms["align_mask"],
            alignment_terms["close_amount"],
            alignment_terms["close_ratio"],
            open_terms["open_dist"],
        )

        step2_reward = (
            alignment_terms["dist_reward"]
            + alignment_terms["quat_reward"]
            + alignment_terms["close_gripper"]
            + open_terms["open_reward"]
            + open_terms["open_delta_reward"]
            + progress_terms["grasp_hold_reward"]
            + progress_terms["open_bonus_reward"]
            + progress_terms["slip_penalty"]
            + penalty_terms["finger_penalty"]
            + penalty_terms["finger_align_reward"]
            + penalty_terms["gripper_switch_penalty"]
        )
        reward = step2_reward + penalty_terms["action_penalty_term"] + penalty_terms["joint_vel_penalty_term"]
        truncation_penalty = np.where(truncated, -reward_cfg.truncation_penalty, 0.0)
        reward = reward + truncation_penalty

        self._update_reward_info(
            state,
            reward_cfg=reward_cfg,
            alignment_terms=alignment_terms,
            open_terms=open_terms,
            progress_terms=progress_terms,
            penalty_terms=penalty_terms,
            gripper_drawer_dist=gripper_drawer_dist,
            truncation_penalty=truncation_penalty,
        )

        return reward

    def _check_termination(self, state: NpEnvState):
        # Check if robot arm extends too far forward causing collision
        robot_grasp_pos_x = self.gripper_tcp.get_pose(state.data)[:, 0]
        drawer_grasp_pos_x = self.drawer_top_handle.get_pose(state.data)[:, 0]
        termination_cfg = self._cfg.termination
        truncated = robot_grasp_pos_x - drawer_grasp_pos_x < termination_cfg.tcp_behind_handle_threshold

        # Check that joint velocity doesn't exceed threshold of 5 rad/s
        joint_vel = self.get_robot_joint_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > termination_cfg.max_joint_vel)
        return truncated

    def get_robot_joint_pos(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_pos(data)[:, : self._num_dof_pos]

    def get_robot_joint_vel(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_vel(data)[:, : self._num_dof_pos]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.robot_default_joint_pos

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel
