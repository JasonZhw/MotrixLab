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

"""Microduck ground-pick task: lower the beak, then return to stand."""

import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.microduck.ground_pick_cfg import ENV_NAME, MicroduckGroundPickNpEnvCfg
from motrix_envs.locomotion.microduck.walk_np import (
    HEAD_SLICE,
    LEG_INDICES,
    NUM_COMMANDS,
    MicroduckWalkTask,
)
from motrix_envs.np.env import NpEnvState


def phase_pose_blend(
    phase: np.ndarray,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> np.ndarray:
    """Return 0=stand / 1=down for descent, hold, rise and rest."""
    phase = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    blend = np.zeros_like(phase)
    descending = phase < descent_end
    blend[descending] = phase[descending] / descent_end
    holding = np.logical_and(phase >= descent_end, phase < hold_end)
    blend[holding] = 1.0
    rising = np.logical_and(phase >= hold_end, phase < rise_end)
    blend[rising] = 1.0 - (phase[rising] - hold_end) / (rise_end - hold_end)
    return blend


def phase_rise_gate(phase: np.ndarray, hold_end: float, rise_end: float) -> np.ndarray:
    """Return a 0..1 gate that stays high during the standing rest."""
    phase = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    gate = np.zeros_like(phase)
    rising = np.logical_and(phase >= hold_end, phase < rise_end)
    gate[rising] = (phase[rising] - hold_end) / (rise_end - hold_end)
    gate[phase >= rise_end] = 1.0
    return gate


def _encode_phase(phase: np.ndarray) -> np.ndarray:
    commands = np.zeros((phase.shape[0], NUM_COMMANDS), dtype=np.float32)
    angle = 2.0 * np.pi * phase
    commands[:, 0] = np.cos(angle)
    commands[:, 1] = np.sin(angle)
    return commands


@registry.env(ENV_NAME, sim_backend="np")
class MicroduckGroundPickTask(MicroduckWalkTask):
    """Bring the mouth tip close to the floor without striking it."""

    def __init__(self, cfg: MicroduckGroundPickNpEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._mouth = self._model.get_site(cfg.asset.mouth_site_name)
        ground = self._model.get_geom_index(cfg.asset.ground_name)
        self._head_contact_pairs = np.asarray(
            [[self._model.get_geom_index(name), ground] for name in cfg.asset.head_collision_names],
            dtype=np.uint32,
        )

    def _sample_commands(self, num_envs: int) -> np.ndarray:
        phase = np.random.uniform(0.0, 1.0, size=num_envs).astype(np.float32)
        return _encode_phase(phase)

    def _update_commands(self, info: dict):
        self._global_step += 1
        info["phase"] = np.mod(
            info["phase"] + self.cfg.ctrl_dt / self.cfg.reward.period,
            1.0,
        ).astype(np.float32)
        angle = 2.0 * np.pi * info["phase"]
        info["commands"][:, 0] = np.cos(angle)
        info["commands"][:, 1] = np.sin(angle)
        info["commands"][:, 2:] = 0.0

    def reset(self, data):
        obs, info = super().reset(data)
        info["phase"] = np.mod(
            np.arctan2(info["commands"][:, 1], info["commands"][:, 0]) / (2.0 * np.pi),
            1.0,
        ).astype(np.float32)
        return obs, info

    def _feet_flat_cost(self, state: NpEnvState) -> np.ndarray:
        costs = []
        for foot in self._feet:
            rotation = foot.get_rotation_mat(state.data)
            local_gravity = np.einsum("nji,j->ni", rotation, self.gravity_vec)
            costs.append(np.sum(np.square(local_gravity[:, :2]), axis=1))
        return np.sum(np.stack(costs, axis=1), axis=1)

    def _head_contacts(self, state: NpEnvState) -> np.ndarray:
        contacts = self._model.get_contact_query(state.data).is_colliding(self._head_contact_pairs)
        return contacts.reshape((self.num_envs, len(self._head_contact_pairs)))

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        values = self._reward_values(state)
        raw = self._reward_terms(state, values)
        weighted = {name: value * self.cfg.reward.scales[name] for name, value in raw.items()}
        reward = np.sum(np.stack(tuple(weighted.values()), axis=0), axis=0).astype(np.float32)

        state.info["Reward"] = weighted
        state.info["metrics"] = {
            "phase": values["phase"].copy(),
            "down_blend": values["down_gate"],
            "return_gate": values["return_gate"],
            "mouth_height": values["mouth_height"],
            "mouth_down_alignment": values["mouth_alignment"],
            "head_contact": values["head_contact"],
            "root_height": values["root_pose"][:, 2],
            "mean_action_abs": np.mean(np.abs(state.info["current_actions"]), axis=1),
        }
        return state.replace(reward=reward)

    def _reward_values(self, state: NpEnvState) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward
        phase = state.info["phase"]
        mouth_height = self._mouth.get_position(state.data)[:, 2]
        mouth_rotation = self._mouth.get_rotation_mat(state.data)
        return {
            "phase": phase,
            "down_gate": phase_pose_blend(phase, cfg.descent_end, cfg.hold_end, cfg.rise_end),
            "return_gate": phase_rise_gate(phase, cfg.hold_end, cfg.rise_end),
            "dof_pos": self.get_dof_pos(state.data),
            "dof_vel": self.get_dof_vel(state.data),
            "lin_vel": self.get_local_linvel(state.data),
            "gyro": self.get_gyro(state.data),
            "gravity": self._projected_gravity(state.data),
            "mouth_height": mouth_height,
            "mouth_alignment": -mouth_rotation[:, 2, 0],
            "head_contact": np.any(self._head_contacts(state), axis=1).astype(np.float32),
            "root_pose": self._body.get_pose(state.data),
        }

    def _reward_terms(
        self, state: NpEnvState, values: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward
        info = state.info
        values = self._reward_values(state) if values is None else values
        phase = values["phase"]
        down_gate = values["down_gate"]
        return_gate = values["return_gate"]
        dof_pos = values["dof_pos"]
        dof_vel = values["dof_vel"]
        lin_vel = values["lin_vel"]
        gyro = values["gyro"]
        gravity = values["gravity"]
        mouth_height = values["mouth_height"]
        mouth_alignment = values["mouth_alignment"]

        leg_pose = np.mean(
            np.exp(
                -np.square(dof_pos[:, LEG_INDICES] - self.default_angles[LEG_INDICES])
                / cfg.leg_return_std**2
            ),
            axis=1,
        )
        neck_pose = np.mean(
            np.exp(
                -np.square(dof_pos[:, HEAD_SLICE] - self.default_angles[HEAD_SLICE])
                / cfg.neck_return_std**2
            ),
            axis=1,
        )
        tilt = np.sum(np.square(gravity[:, :2]), axis=1)
        return {
            "mouth_ground_proximity": down_gate
            * np.exp(-np.square(mouth_height) / cfg.mouth_height_std**2),
            "mouth_perpendicular": down_gate * np.clip(mouth_alignment, -1.0, 1.0),
            "return_pose_legs": return_gate * leg_pose,
            "return_pose_neck": return_gate * neck_pose,
            "return_upright": return_gate * np.exp(-tilt / cfg.return_upright_std**2),
            "upright": np.exp(-tilt / cfg.always_upright_std**2),
            "feet_grounded": np.mean(info["contacts"], axis=1),
            "feet_flat": self._feet_flat_cost(state),
            "head_contact": values["head_contact"],
            "neck_velocity_descent": (phase < cfg.hold_end).astype(np.float32)
            * np.mean(np.square(dof_vel[:, HEAD_SLICE]), axis=1),
            "horizontal_velocity": np.sum(np.square(lin_vel[:, :2]), axis=1),
            "vertical_velocity": np.square(lin_vel[:, 2]),
            "body_ang_vel": np.sum(np.square(gyro[:, :2]), axis=1),
            "action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "leg_action_rate": np.mean(
                np.square(
                    info["current_actions"][:, LEG_INDICES]
                    - info["last_actions"][:, LEG_INDICES]
                ),
                axis=1,
            ),
            "joint_velocity": np.sum(np.square(dof_vel), axis=1),
            "joint_acceleration": np.sum(
                np.square((dof_vel - info["last_dof_vel"]) / self.cfg.ctrl_dt), axis=1
            ),
            "joint_limits": self._joint_limit_cost(dof_pos),
            "termination": state.terminated.astype(np.float32),
        }
