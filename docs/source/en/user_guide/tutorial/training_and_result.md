# Training Execution and Result Analysis

This section introduces how to execute reinforcement learning training and how to analyze and use training results.

## Start Training

### Basic Training Commands

```bash
# Train with default parameters (SKRL framework)
uv run scripts/train.py --env cartpole

# Specify RL framework
uv run scripts/train.py --env cartpole --rllib skrl
uv run scripts/train.py --env cartpole --rllib rslrl

# Specify simulation backend
uv run scripts/train.py --env cartpole --sim-backend np

# Specify training backend (SKRL only)
uv run scripts/train.py --env cartpole --rllib skrl --train-backend jax
uv run scripts/train.py --env cartpole --rllib skrl --train-backend torch
```

### Advanced Training Configuration

```bash
# Customize training parameters with SKRL
uv run scripts/train.py --env cartpole \
  --rllib skrl \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# Customize training parameters with RSLRL
uv run scripts/train.py --env cartpole \
  --rllib rslrl \
  --num-envs 1024 \
  --sim-backend np

# Note: Parameters like learning rate need to be set through configuration files or code override

# Enable rendering to monitor training process
uv run scripts/train.py --env cartpole --render
```

### Different Framework Configuration

The system supports different RL frameworks with different configuration systems:

-   **SKRL Framework**: Supports JAX and PyTorch training backends with configurable parameters per backend (via Python dataclasses)
-   **RSLRL Framework**: Supports PyTorch backend with configuration via Python dataclasses (RslrlCfg)

For SKRL, the system supports configuring different reinforcement learning parameters for different training backends (JAX/Torch).

### Supported Command Line Parameters

| Parameter         | Description                             | Default Value |
| ----------------- | --------------------------------------- | ------------- |
| `--env`           | Environment name                        | `cartpole`    |
| `--rllib`         | RL framework (skrl/rslrl)               | `skrl`        |
| `--sim-backend`   | Simulation backend (np)                 | Auto select   |
| `--train-backend` | Training backend (jax/torch, SKRL only) | Auto select   |
| `--num-envs`      | Number of parallel environments         | 2048          |
| `--render`        | Enable rendering                        | False         |

> **Note**: Other parameters such as learning rate, network structure, etc., need to be set through separate configuration files.

## Training Process Monitoring

### TensorBoard Monitoring

Start TensorBoard to view training progress:

```bash
uv run tensorboard --logdir runs/{env-name}
```

For example:

```bash
uv run tensorboard --logdir runs/cartpole
```

## Model Evaluation and Testing

### Using Trained Policies

```bash
# Automatically find best policy for testing (recommended)
uv run scripts/play.py --env cartpole

# Manually specify policy file for testing
uv run scripts/play.py --env cartpole --policy runs/cartpole/nn/best_agent.pickle

# Specify number of test environments
uv run scripts/play.py --env cartpole --num-envs 100
```

> **Note**: The system will automatically find the latest and best policy files in the `runs/cartpole/` directory for testing.
