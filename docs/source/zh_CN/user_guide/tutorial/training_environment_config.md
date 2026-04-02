# 训练环境配置

MotrixLab 提供了灵活的配置系统，允许用户自定义强化学习训练参数。本节介绍如何配置训练环境和强化学习算法参数。

## RL 训练配置

MotrixLab 支持多个 RL 框架，具有不同的配置系统：

-   **SKRL 框架**：使用 Python 数据类配置（SkrlCfg）
-   **RSLRL 框架**：使用 Python 数据类配置（RslrlCfg）

### SKRL 配置 (SkrlCfg)

训练配置定义了基于 PPO 算法的强化学习算法的参数。MotrixLab 现在支持为不同训练后端配置不同的参数。

#### 完整配置示例

以下是 `CartPoleSkrlPpo` 的实际配置，展示了完整的显式参数填充方法。该配置使用较小的网络 `[32, 32]`，适合 CartPole 这样的简单任务。

```{literalinclude} ../../../../motrix_rl/src/motrix_rl/tasks/cartpole.py
:language: python
:start-after: docs-start: cartpole-skrl-config
:end-before: docs-end: cartpole-skrl-config
```

**关键配置说明：**

-   **网络架构**: `hiddens=[32, 32]` - CartPole 是简单任务，使用小网络即可（默认: `[256, 128, 64]`）
-   **训练轮数**: `learning_epochs=5` - 比默认值 2 更高，确保充分学习
-   **小批量数量**: `mini_batches=4` - 比默认值 32 更少，适合简单任务
-   **训练时长**: `timesteps=5000` - 对于 CartPole 来说已经足够（默认: 10000）
-   **所有参数**: 从父类继承的所有参数都被显式指定，无隐藏默认值

完整的源代码请参考: [`motrix_rl/src/motrix_rl/tasks/cartpole.py`](https://github.com/Motphys/motrix-lab/blob/main/motrix_rl/src/motrix_rl/tasks/cartpole.py)

### RSLRL 配置 (RslrlCfg)

RSLRL 是另一个高性能强化学习库，专门用于四足机器人等复杂控制任务。

#### 完整配置示例

以下是 `CartPoleRslrlPpo` 的实际配置，展示了完整的显式参数填充方法。该配置使用较小的网络 `[32, 32]`，适合 CartPole 这样的简单任务。

```{literalinclude} ../../../../motrix_rl/src/motrix_rl/tasks/cartpole.py
:language: python
:start-after: docs-start: cartpole-rslrl-config
:end-before: docs-end: cartpole-rslrl-config
```

**关键配置说明：**

-   **网络架构**: `hidden_dims=[32, 32]` - CartPole 是简单任务，使用小网络即可（默认: `[256, 128, 64]`）
-   **训练迭代**: `max_iterations=300` - 总共训练 300 次迭代
-   **每轮步数**: `num_steps_per_env=16` - 每个环境收集 16 步
-   **学习率**: `learning_rate=5.0e-4` - 学习率设置
-   **熵系数**: `entropy_coef=5e-3` - 熵系数，用于探索
-   **所有参数**: 从父类继承的所有参数都被显式指定，无隐藏默认值

详细的 RSLRL 配置选项和默认值，请参考：

-   `motrix_rl/rslrl/cfg.py`：配置类定义
-   `motrix_rl/template/rslrl_config.yaml`：YAML 参考模板

## 配置使用方法

### 1. 默认配置使用

```bash
# 使用代码中给定的配置（默认：SKRL 框架）
uv run scripts/train.py --env my-task

# 指定 RL 框架
uv run scripts/train.py --env my-task --rllib skrl
uv run scripts/train.py --env my-task --rllib rslrl

# 指定 SKRL 的训练后端，系统会自动选择对应的后端配置
uv run scripts/train.py --env my-task --rllib skrl --train-backend jax
uv run scripts/train.py --env my-task --rllib skrl --train-backend torch
```

### 2. 命令行参数覆盖

```bash
# 覆盖支持的命令行参数
uv run scripts/train.py --env my-task \
  --rllib skrl \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# 系统会自动选择JAX后端对应的配置
```
