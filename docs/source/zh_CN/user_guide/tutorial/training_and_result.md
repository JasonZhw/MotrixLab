# 训练执行和结果分析

本节介绍如何执行强化学习训练，以及如何分析和使用训练结果。

## 启动训练

### 基本训练命令

```bash
# 使用默认参数训练（SKRL 框架）
uv run scripts/train.py --env cartpole

# 指定 RL 框架
uv run scripts/train.py --env cartpole --rllib skrl
uv run scripts/train.py --env cartpole --rllib rslrl

# 指定仿真后端
uv run scripts/train.py --env cartpole --sim-backend np

# 指定训练后端（仅 SKRL）
uv run scripts/train.py --env cartpole --rllib skrl --train-backend jax
uv run scripts/train.py --env cartpole --rllib skrl --train-backend torch
```

### 高级训练配置

```bash
# 使用 SKRL 自定义训练参数
uv run scripts/train.py --env cartpole \
  --rllib skrl \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np

# 使用 RSLRL 自定义训练参数
uv run scripts/train.py --env cartpole \
  --rllib rslrl \
  --num-envs 1024 \
  --sim-backend np

# 注意：学习率等参数需要通过配置文件或代码覆盖设置

# 启用渲染监控训练过程
uv run scripts/train.py --env cartpole --render
```

### 不同框架配置

系统支持不同的 RL 框架，具有不同的配置系统：

-   **SKRL 框架**：支持 JAX 和 PyTorch 训练后端，可通过 Python 数据类为每个后端配置参数
-   **RSLRL 框架**：支持 PyTorch 后端，通过 Python 数据类（RslrlCfg）进行配置

对于 SKRL，系统支持为不同训练后端（JAX/Torch）配置不同的强化学习参数。例如：

### 支持的命令行参数

| 参数              | 说明                          | 默认值     |
| ----------------- | ----------------------------- | ---------- |
| `--env`           | 环境名称                      | `cartpole` |
| `--rllib`         | RL 框架 (skrl/rslrl)          | `skrl`     |
| `--sim-backend`   | 仿真后端 (np)                 | 自动选择   |
| `--train-backend` | 训练后端 (jax/torch，仅 SKRL) | 自动选择   |
| `--num-envs`      | 并行环境数量                  | 2048       |
| `--render`        | 启用渲染                      | False      |

> **注意**: 其他参数如学习率、网络结构等需要通过单独文件设置。

## 训练过程监控

### TensorBoard 监控

启动 TensorBoard 查看训练进度：

```bash
uv run tensorboard --logdir runs/{env-name}
```

例如：

```bash
uv run tensorboard --logdir runs/cartpole
```

## 模型评估和测试

### 使用训练好的策略

```bash
# 自动寻找最佳策略测试（推荐）
uv run scripts/play.py --env cartpole

# 手动指定策略文件测试
uv run scripts/play.py --env cartpole --policy runs/cartpole/nn/best_agent.pickle

# 指定测试环境数量
uv run scripts/play.py --env cartpole --num-envs 100
```

> **说明**：系统会自动在 `runs/cartpole/` 目录下寻找最新、最佳的策略文件进行测试。
