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

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from skrl.agents.jax.ppo import PPO as BasePPO
from skrl.envs.jax import Wrapper
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl import registry, utils
from motrix_rl.skrl import get_log_dir
from motrix_rl.skrl.config import SkrlCfg, SkrlMemoryCfg
from motrix_rl.skrl.jax import wrap_env


def _instantiate_memory(memory_cfg: SkrlMemoryCfg, memory_size: int, num_envs: int, device) -> Any:
    """Instantiate a SKRL Memory class based on configuration.

    Args:
        memory_cfg: Memory configuration with class_name and settings
        memory_size: Size of the memory buffer
        num_envs: Number of parallel environments
        device: Device to place memory on

    Returns:
        Instantiated SKRL Memory object

    Raises:
        ValueError: If class_name is not supported
    """
    from skrl.memories.jax import RandomMemory

    # Map class_name to actual Memory class
    memory_classes = {
        "RandomMemory": RandomMemory,
    }

    class_name = memory_cfg.class_name
    if class_name not in memory_classes:
        raise ValueError(f"Unsupported memory class_name: {class_name}. Supported: {list(memory_classes.keys())}")

    MemoryClass = memory_classes[class_name]
    return MemoryClass(memory_size=memory_size, num_envs=num_envs, device=device)


def _add_runtime_config(
    cfg: dict,
    env: Wrapper,
    log_dir: str = None,
) -> dict:
    """Add runtime-specific configuration to the base agent config.

    Args:
        cfg: Base configuration from agent.to_dict() (will be modified in-place)
        env: SKRL environment wrapper
        log_dir: Optional logging directory path

    Returns:
        The same cfg dict with runtime values added (modified in-place for convenience)
    """
    # Convert learning_rate_scheduler from string to actual class (if configured)
    if cfg.get("learning_rate_scheduler") == "KLAdaptiveLR":
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
    # Otherwise keep as-is (None or other scheduler type)

    # Add rewards shaper (conditional based on rewards_shaper_scale in cfg)
    if cfg.get("rewards_shaper_scale", 1.0) != 1.0:
        cfg["rewards_shaper"] = lambda reward, timestep, timesteps: reward * cfg["rewards_shaper_scale"]
    else:
        cfg["rewards_shaper"] = None

    # Add preprocessors (require runtime env values)
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    # Add experiment configuration (handle -1 -> "auto" conversion)
    if log_dir:
        cfg["experiment"]["write_interval"] = (
            "auto" if cfg["experiment"]["write_interval"] == -1 else cfg["experiment"]["write_interval"]
        )
        cfg["experiment"]["checkpoint_interval"] = (
            "auto" if cfg["experiment"]["checkpoint_interval"] == -1 else cfg["experiment"]["checkpoint_interval"]
        )
        cfg["experiment"]["directory"] = log_dir
    else:
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0

    return cfg


class PPO(BasePPO):
    _total_custom_rewards: dict[str, np.ndarray] = {}

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ) -> None:
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

        if "Reward" in infos:
            for key, value in infos["Reward"].items():
                self.tracking_data[f"Reward Instant / {key} (max)"].append(jnp.max(value))
                self.tracking_data[f"Reward Instant / {key} (min)"].append(jnp.min(value))
                self.tracking_data[f"Reward Instant / {key} (mean)"].append(jnp.mean(value))
                if key not in self._total_custom_rewards:
                    self._total_custom_rewards[key] = jnp.zeros_like(value)
                self._total_custom_rewards[key] += value
            done = terminated | truncated
            done = done.reshape(-1)
            if done.any():
                for key in self._total_custom_rewards:
                    self.tracking_data[f"Reward Total/ {key} (mean)"].append(
                        jnp.mean(self._total_custom_rewards[key][done])
                    )
                    self.tracking_data[f"Reward Total/ {key} (min)"].append(
                        jnp.min(self._total_custom_rewards[key][done])
                    )
                    self.tracking_data[f"Reward Total/ {key} (max)"].append(
                        jnp.max(self._total_custom_rewards[key][done])
                    )

                    self._total_custom_rewards[key] = self._total_custom_rewards[key] * (1 - done)

        if "metrics" in infos:
            for key, value in infos["metrics"].items():
                self.tracking_data[f"metrics / {key} (max)"].append(jnp.max(value))
                self.tracking_data[f"metrics / {key} (min)"].append(jnp.min(value))
                self.tracking_data[f"metrics / {key} (mean)"].append(jnp.mean(value))


class Trainer:
    _trainer: SequentialTrainer
    _env_name: str
    _sim_backend: str
    _rlcfg: SkrlCfg
    _enable_render: bool

    def __init__(
        self,
        env_name: str,
        sim_backend: str = None,
        enable_render: bool = False,
        cfg_override: dict = None,
    ) -> None:
        rlcfg = registry.default_rl_cfg(env_name, "skrl", backend="jax")
        if cfg_override is not None:
            rlcfg = utils.cfg_override(rlcfg, cfg_override)
        self._rlcfg = rlcfg
        self._env_name = env_name
        self._sim_backend = sim_backend
        self._enable_render = enable_render

    def train(self) -> None:
        """
        Start training the agent.
        """
        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.num_envs)

        set_seed(rlcfg.runner.seed)
        skrl_env = wrap_env(env, self._enable_render)
        models = self._make_model(skrl_env, rlcfg)
        # Get base configuration from config object
        ppo_cfg = rlcfg.runner.agent.to_dict()
        # Add runtime-specific configuration
        _add_runtime_config(ppo_cfg, skrl_env, log_dir=get_log_dir(self._env_name, rllib="skrl", agent_name="PPO"))
        agent = self._make_agent(models, skrl_env, ppo_cfg, rlcfg.runner.memory)
        cfg_trainer = {
            "timesteps": rlcfg.runner.trainer.timesteps,
            "headless": not self._enable_render,
        }
        trainer = SequentialTrainer(cfg=cfg_trainer, env=skrl_env, agents=agent)
        trainer.train()

    def play(self, policy: str) -> None:
        import time

        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.play_num_envs)

        set_seed(rlcfg.runner.seed)
        env = wrap_env(env, self._enable_render)
        models = self._make_model(env, rlcfg)
        # Get base configuration from config object
        ppo_cfg = rlcfg.runner.agent.to_dict()
        # Add runtime-specific configuration
        _add_runtime_config(ppo_cfg, env)
        agent = self._make_agent(models, env, ppo_cfg, rlcfg.runner.memory)
        agent.load(policy)
        obs, _ = env.reset()

        fps = 60
        while True:
            t = time.time()
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
            env.render()
            delta_time = time.time() - t
            if delta_time < 1.0 / fps:
                time.sleep(1.0 / fps - delta_time)

    def _make_model(self, env: Wrapper, rlcfg: SkrlCfg) -> dict[str, Model]:
        _activation_fn = {
            "elu": nn.elu,
            "relu": nn.relu,
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
            "leaky_relu": nn.leaky_relu,
            "selu": nn.selu,
        }

        policy_cfg = rlcfg.runner.models.policy
        value_cfg = rlcfg.runner.models.value

        def resolve_activations(activation_names: list[str], hiddens: list[int]) -> list:
            if len(activation_names) == 1:
                return [_activation_fn[activation_names[0]]] * len(hiddens)
            if len(activation_names) != len(hiddens):
                raise ValueError(
                    f"hidden_activation length ({len(activation_names)}) must be 1 or "
                    f"match hiddens length ({len(hiddens)})"
                )
            return [_activation_fn[name] for name in activation_names]

        policy_acts = resolve_activations(policy_cfg.hidden_activation, policy_cfg.hiddens)
        value_acts = resolve_activations(value_cfg.hidden_activation, value_cfg.hiddens)

        class Policy(GaussianMixin, Model):
            def __init__(self, observation_space, action_space, device=None, **kwargs):
                Model.__init__(self, observation_space, action_space, device, **kwargs)
                GaussianMixin.__init__(
                    self,
                    policy_cfg.clip_actions,
                    policy_cfg.clip_log_std,
                    policy_cfg.min_log_std,
                    policy_cfg.max_log_std,
                    policy_cfg.reduction,
                )

            @nn.compact
            def __call__(self, inputs, role):
                x = inputs["states"]
                for size, act in zip(policy_cfg.hiddens, policy_acts):
                    x = act(nn.Dense(size)(x))
                x = nn.Dense(self.num_actions)(x)
                log_std = self.param("log_std", lambda _: jnp.full(self.num_actions, float(policy_cfg.initial_log_std)))
                return x, log_std, {}

        class Value(DeterministicMixin, Model):
            def __init__(self, observation_space, action_space, device=None, **kwargs):
                Model.__init__(self, observation_space, action_space, device, **kwargs)
                DeterministicMixin.__init__(self, value_cfg.clip_actions)

            @nn.compact
            def __call__(self, inputs, role):
                x = inputs["states"]
                for size, act in zip(value_cfg.hiddens, value_acts):
                    x = act(nn.Dense(size)(x))
                x = nn.Dense(1)(x)
                return x, {}

        models = {}
        models["policy"] = Policy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        models["value"] = Value(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )

        for role, model in models.items():
            model.init_state_dict(role)

        return models

    def _make_agent(
        self, models: dict[str, Model], env: Wrapper, ppo_cfg: dict[str, Any], memory_cfg: SkrlMemoryCfg
    ) -> PPO:
        # Use memory_size from SkrlMemoryCfg, fall back to rollouts if -1
        memory_size = memory_cfg.memory_size
        if memory_size == -1:
            memory_size = ppo_cfg["rollouts"]

        memory = _instantiate_memory(memory_cfg, memory_size, env.num_envs, env.device)

        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        return agent


class Player:
    def __init__(self, env_name: str, sim_backend: str = None) -> None:
        pass
