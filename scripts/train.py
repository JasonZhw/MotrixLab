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

from absl import app, flags
from skrl import config

from motrix_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "The env to train")
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to train")
_RENDER = flags.DEFINE_bool("render", False, "Render the env")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", None, "The learning backend. (jax/torch)")
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")
_RLLIB = flags.DEFINE_string("rllib", "skrl", "The RL framework (skrl/rslrl)")


def get_train_backend(supports: utils.DeviceSupports, train_backend_arg: str | None, rllib: str):
    """
    Determine the training backend based on device supports, user input, and RL framework.

    Args:
        supports: Device support information
        train_backend_arg: User-specified backend via --train-backend flag (None if not provided)
        rllib: RL framework to use ("skrl" or "rslrl")

    Returns:
        The determined backend name ("jax" or "torch")

    Raises:
        Exception: If user specifies incompatible backend or no backend is available
    """
    # RSLRL only supports PyTorch
    if rllib == "rslrl":
        if train_backend_arg is not None and train_backend_arg != "torch":
            raise Exception("RSLRL only supports PyTorch backend.")
        if not supports.torch:
            raise Exception("RSLRL requires PyTorch, but it is not available on your device.")
        return "torch"

    # User explicitly specified backend
    if train_backend_arg is not None:
        backend = train_backend_arg
        if backend == "jax" and not supports.jax:
            raise Exception("JAX is not available on your device.")
        if backend == "torch" and not supports.torch:
            raise Exception("PyTorch is not available on your device.")
        return backend

    # Auto-select backend based on device priority
    if supports.jax and supports.jax_gpu:
        return "jax"
    elif supports.torch and supports.torch_gpu:
        return "torch"
    elif supports.jax:
        return "jax"
    elif supports.torch:
        return "torch"
    else:
        raise Exception("Neither JAX nor PyTorch is available on the device.")


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    env_name = _ENV.value
    enable_render = _RENDER.value

    rl_override = {}

    if _NUM_ENVS.present:
        rl_override["num_envs"] = _NUM_ENVS.value

    if _RAND_SEED.value:
        rl_override["runner.seed"] = None
    elif _SEED.present:
        rl_override["runner.seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value
    rllib = _RLLIB.value

    # Determine the training backend
    train_backend = get_train_backend(device_supports, _TRAIN_BACKEND.value, rllib)

    trainer = None
    if rllib == "rslrl":
        # RSLRL training flow
        assert device_supports.torch, "PyTorch is not available on your device"
        assert train_backend == "torch", "RSLRL only supports PyTorch backend"
        from motrix_rl.rslrl.torch.train import ppo

        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)

    elif train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # or "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)

    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo

        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
    else:
        raise Exception(f"Unknown train backend: {train_backend}")

    trainer.train()


if __name__ == "__main__":
    app.run(main)
