# Training Environment Configuration

MotrixLab provides a flexible configuration system that allows users to customize reinforcement learning training parameters. This section introduces how to configure training environments and reinforcement learning algorithm parameters.

## RL Training Configuration

MotrixLab supports multiple RL frameworks with different configuration systems:

-   **SKRL Framework**: Uses Python dataclass configuration (SkrlCfg)
-   **RSLRL Framework**: Uses Python dataclass configuration (RslrlCfg)

### SKRL Configuration (SkrlCfg)

Training configuration defines parameters for reinforcement learning algorithms based on PPO. MotrixLab now supports configuring different parameters for different training backends.

#### Complete Configuration Example

The following is the actual `CartPoleSkrlPpo` configuration, demonstrating complete explicit parameter filling. This configuration uses a smaller network `[32, 32]` suitable for simple tasks like CartPole.

```{literalinclude} ../../../../motrix_rl/src/motrix_rl/tasks/cartpole.py
:language: python
:start-after: docs-start: cartpole-skrl-config
:end-before: docs-end: cartpole-skrl-config
```

**Key Configuration Notes:**

-   **Network Architecture**: `hiddens=[32, 32]` - CartPole is a simple task, a small network is sufficient (default: `[256, 128, 64]`)
-   **Training Epochs**: `learning_epochs=5` - Higher than default value 2, ensuring thorough learning
-   **Mini-batches**: `mini_batches=4` - Fewer than default 32, suitable for simple tasks
-   **Training Duration**: `timesteps=5000` - Sufficient for CartPole (default: 10000)
-   **All Parameters**: All parameters from parent classes are explicitly specified, no hidden defaults

For complete source code, see: [`motrix_rl/src/motrix_rl/tasks/cartpole.py`](https://github.com/Motphys/motrix-lab/blob/main/motrix_rl/src/motrix_rl/tasks/cartpole.py)

### RSLRL Configuration (RslrlCfg)

RSLRL is another high-performance reinforcement learning library, specifically designed for complex control tasks like quadruped robots.

#### Complete Configuration Example

The following is the actual `CartPoleRslrlPpo` configuration, demonstrating complete explicit parameter filling. This configuration uses a smaller network `[32, 32]` suitable for simple tasks like CartPole.

```{literalinclude} ../../../../motrix_rl/src/motrix_rl/tasks/cartpole.py
:language: python
:start-after: docs-start: cartpole-rslrl-config
:end-before: docs-end: cartpole-rslrl-config
```

**Key Configuration Notes:**

-   **Network Architecture**: `hidden_dims=[32, 32]` - CartPole is a simple task, a small network is sufficient (default: `[256, 128, 64]`)
-   **Training Iterations**: `max_iterations=300` - Total of 300 training iterations
-   **Steps per Environment**: `num_steps_per_env=16` - Number of steps to collect per environment
-   **Learning Rate**: `learning_rate=5.0e-4` - Learning rate setting
-   **Entropy Coefficient**: `entropy_coef=5e-3` - Entropy coefficient for exploration
-   **All Parameters**: All parameters from parent classes are explicitly specified, no hidden defaults

For detailed RSLRL configuration options and default values, refer to:

-   `motrix_rl/rslrl/cfg.py`: Configuration class definitions
-   `motrix_rl/template/rslrl_config.yaml`: YAML reference template

## Configuration Usage Methods

### 1. Default Configuration Usage

```bash
# Use configuration given in code (default: SKRL framework)
uv run scripts/train.py --env my-task

# Specify RL framework
uv run scripts/train.py --env my-task --rllib skrl
uv run scripts/train.py --env my-task --rllib rslrl

# Specify training backend for SKRL, system will automatically select corresponding backend configuration
uv run scripts/train.py --env my-task --rllib skrl --train-backend jax
uv run scripts/train.py --env my-task --rllib skrl --train-backend torch
```

### 2. Command Line Parameter Override

```bash
# Override supported command line parameters
uv run scripts/train.py --env my-task \
  --rllib skrl \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# System will automatically select JAX backend configuration
```
