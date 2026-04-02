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

"""PPO Trainer for RSLRL integration."""

import logging

import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl import utils
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap
from motrix_rl.skrl import get_log_dir

logger = logging.getLogger(__name__)


class Trainer:
    """RSLRL PPO Trainer.

    This class wraps RSLRL's OnPolicyRunner to provide a training interface
    consistent with the SKRL trainer implementation.
    """

    _env_name: str
    _sim_backend: str
    _rlcfg: RslrlCfg
    _enable_render: bool

    def __init__(
        self,
        env_name: str,
        sim_backend: str = None,
        enable_render: bool = False,
        cfg_override: dict = None,
    ) -> None:
        """Initialize the RSLRL PPO trainer.

        Args:
            env_name: Name of the environment to train
            sim_backend: Simulation backend to use (e.g., "mujoco", "npcm")
            enable_render: Whether to enable rendering during training
            cfg_override: Optional configuration overrides
        """
        rlcfg = rl_registry.default_rl_cfg(env_name, "rslrl", backend="torch")
        if cfg_override is not None:
            rlcfg = utils.cfg_override(rlcfg, cfg_override)
        self._rlcfg = rlcfg
        self._env_name = env_name
        self._sim_backend = sim_backend
        self._enable_render = enable_render

    def train(self) -> None:
        """Start training the agent.

        Creates the environment, wraps it for RSLRL, and runs the training loop.
        """
        rlcfg = self._rlcfg

        # Create environment
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.num_envs)

        # Set random seed
        if rlcfg.runner.seed is not None:
            torch.manual_seed(rlcfg.runner.seed)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Wrap environment for RSLRL
        vec_env = RslrlNpEnvWrap(env, device)

        # Create RSLRL config - use to_dict() method
        rslrl_cfg = self._create_rslrl_config()

        # Create RSLRL runner
        runner = OnPolicyRunner(
            vec_env, rslrl_cfg, log_dir=get_log_dir(self._env_name, rllib="rslrl", agent_name="PPO"), device=device
        )

        # Start training
        logger.info(f"Starting training for {self._env_name}")
        logger.info(f"Number of environments: {rlcfg.num_envs}")

        # Get max_iterations from config
        total_iterations = rslrl_cfg["max_iterations"]
        logger.info(f"Number of learning iterations: {total_iterations}")

        runner.learn(num_learning_iterations=total_iterations)

        logger.info("Training completed")

    def play(self, policy_path: str) -> None:
        """Evaluate a trained policy.

        Args:
            policy_path: Path to the saved policy file
        """
        import time

        rlcfg = self._rlcfg

        # Create environment with play_num_envs
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.play_num_envs)

        # Set random seed
        if rlcfg.runner.seed is not None:
            torch.manual_seed(rlcfg.runner.seed)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Wrap environment for RSLRL
        vec_env = RslrlNpEnvWrap(env, device)

        # Create RSLRL config (minimal for evaluation)
        rslrl_cfg = self._create_rslrl_config()

        # Create RSLRL runner with log_dir=None to disable logging (no git diff storage in play mode)
        runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=device)

        # Load policy
        logger.info(f"Loading policy from {policy_path}")
        runner.load(policy_path)

        # Run evaluation loop
        logger.info("Starting evaluation loop...")
        logger.info("Press Ctrl+C to stop")
        obs, _ = vec_env.reset()
        fps = 60

        try:
            while True:
                t = time.time()

                # Get actions from policy
                with torch.no_grad():
                    policy = runner.get_inference_policy(device=device)
                    # MLPModel is callable, returns distribution mean for deterministic evaluation
                    actions = policy(obs)

                # Step environment
                obs, rewards, dones, infos = vec_env.step(actions)

                # Render the environment
                vec_env.render()

                delta_time = time.time() - t
                if delta_time < 1.0 / fps:
                    time.sleep(1.0 / fps - delta_time)

        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")

    def _create_rslrl_config(self) -> dict:
        return self._rlcfg.runner.to_dict()
