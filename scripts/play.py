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

import logging
from pathlib import Path

from absl import app, flags
from skrl import config

from motrix_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "The env to play")
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)
_POLICY = flags.DEFINE_string("policy", None, "The policy to load")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to play")
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")
_RLLIB = flags.DEFINE_string(
    "rllib", None, "The RL framework (skrl/rslrl). Auto-discovered from latest training if not specified."
)


def get_inference_backend(policy_path: Path | str, rllib: str):
    """Determine the backend from RL framework and policy file extension."""
    if rllib == "rslrl":
        # RSLRL always uses torch backend
        return "torch"
    # Handle both Path and str types
    suffix = policy_path.suffix if isinstance(policy_path, Path) else Path(policy_path).suffix
    if suffix == ".pt":
        return "torch"
    if suffix == ".pickle":
        return "jax"
    else:
        raise Exception(f"Unknown policy format: {policy_path}")


def discover_rllib(env_name: str) -> tuple[str, Path]:
    """
    Discover the RL framework and best policy from the most recent training run.

    Args:
        env_name: The name of the environment

    Returns:
        Tuple of (RL framework name, path to best policy)

    Raises:
        FileNotFoundError: If no training results are found
    """
    base_dir = Path(f"runs/{env_name}")

    if not base_dir.exists():
        raise FileNotFoundError(f"No training results found for environment '{env_name}' in {base_dir}")

    frameworks = []
    for framework in ["skrl", "rslrl"]:
        framework_dir = base_dir / framework
        if framework_dir.exists() and framework_dir.is_dir():
            # Get all training run directories
            training_runs = [d for d in framework_dir.iterdir() if d.is_dir()]
            if training_runs:
                # Find the most recent run for this framework
                latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
                frameworks.append((framework, latest_run.stat().st_mtime, latest_run))

    if not frameworks:
        raise FileNotFoundError(f"No training runs found for environment '{env_name}' in {base_dir}")

    # Return the framework with the most recent training run and its best policy
    latest_framework, _, latest_run_dir = max(frameworks, key=lambda x: x[1])
    logger.info(f"Auto-discovered RL framework: {latest_framework}")

    # Find best policy in the latest run directory
    if latest_framework == "rslrl":
        # RSLRL uses model_*.pt format
        model_files = list(latest_run_dir.glob("model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No policy files found in {latest_run_dir}")

        def extract_iteration(filename):
            stem = Path(filename).stem
            parts = stem.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
            return 0

        best_policy = max(model_files, key=lambda f: (f.stat().st_mtime, extract_iteration(f)))
    else:
        # SKRL uses checkpoints subdirectory
        checkpoints_dir = latest_run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found in {latest_run_dir}")

        # First, try to find best_agent files
        best_files = list(checkpoints_dir.glob("best_agent.*"))
        if best_files:
            best_policy = best_files[0]
        else:
            # Find checkpoint with highest timestep
            checkpoint_files = list(checkpoints_dir.glob("agent_*.pt")) + list(checkpoints_dir.glob("agent_*.pickle"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No policy files found in {checkpoints_dir}")

            def extract_timestep(filename):
                stem = Path(filename).stem
                parts = stem.split("_")
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return 0
                return 0

            best_policy = max(checkpoint_files, key=extract_timestep)

    return latest_framework, best_policy


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    env_name = _ENV.value
    enable_render = True

    rl_override = {}

    if _NUM_ENVS.present:
        rl_override["play_num_envs"] = _NUM_ENVS.value

    if _RAND_SEED.value:
        rl_override["runner.seed"] = None
    elif _SEED.present:
        rl_override["runner.seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value
    rllib = None
    policy_path = None

    if _POLICY.present:
        if not _RLLIB.present:
            logger.error("Error: --policy specified but --rllib not specified")
            return
        rllib = _RLLIB.value
        policy_path = _POLICY.value
        logger.info(f"Using specified RL framework: {rllib}")
        logger.info(f"Using specified policy: {policy_path}")
    else:
        # if policy is not specified, search for the lastest training run and use its best policy
        try:
            rllib, policy_path = discover_rllib(env_name)
            logger.info(f"Auto-discovered RL framework: {rllib}")
            logger.info(f"Auto-discovered best policy: {policy_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("Please specify --rllib or train a model first")
            return

    backend = get_inference_backend(policy_path, rllib)

    if rllib == "rslrl":
        # RSLRL evaluation flow (always uses torch backend)
        assert device_supports.torch, "PyTorch is not available on your device"
        from motrix_rl.rslrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)

    elif backend == "jax":
        assert device_supports.jax, "jax is not avaliable on your device "
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # or "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)

    elif backend == "torch":
        assert device_supports.torch, "torch is not avaliable on your device"
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)


if __name__ == "__main__":
    app.run(main)
