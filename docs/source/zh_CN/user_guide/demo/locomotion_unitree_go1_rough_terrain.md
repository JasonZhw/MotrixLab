# Unitree GO1 复杂地形行走

Unitree GO1 复杂地形行走环境是一个四足机器人强化学习任务，旨在训练机器人在具有挑战性的地形上实现稳定行走。该环境包含两种主要的地形类型：粗糙地形和台阶地形。

```{video} /_static/videos/go1_rough_terrain_walk.mp4
:poster: _static/images/poster/go1_rough_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

```{video} /_static/videos/go1_stairs_terrain_walk.mp4
:poster: _static/images/poster/go1_stairs_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

训练 Unitree GO1 四足机器人在复杂地形上实现稳定、高效的双足行走。该环境使用 MotrixSim 物理引擎进行仿真，提供高保真的动力学模拟。智能体通过控制各关节的目标位置来实现速度跟踪和姿态稳定，同时适应不同的地形挑战。

### 任务目标

-   **速度跟踪**：准确跟踪给定的线速度和角速度命令
-   **姿态稳定**：在各种地形条件下保持身体姿态稳定
-   **能量效率**：以最小的能耗实现行走任务
-   **地形适应性**：适应粗糙地形和台阶地形的不同挑战

---

## 动作空间（Action Space）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-1.0, 1.0, (12,), float32)` |
| **维度** | 12                               |

动作对应 12 个关节的位置控制指令，包括四条腿的髋关节、大腿关节和小腿关节。

---

## 观察空间

### 粗糙地形观察空间（48 维）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (48,), float32)` |
| **维度** | 48                               |

| 部分                  | 内容说明                 | 维度 | 备注           |
| --------------------- | ------------------------ | ---- | -------------- |
| **noisy_linvel**      | 机体坐标系下的线速度     | 3    | 带噪声的线速度 |
| **noisy_gyro**        | 机体坐标系下的角速度     | 3    | 带噪声的角速度 |
| **local_gravity**     | 局部坐标系下的重力方向   | 3    | 重力向量投影   |
| **noisy_joint_angle** | 关节角度与默认角度的偏差 | 12   | 12 个关节      |
| **noisy_joint_vel**   | 关节角速度               | 12   | 带噪声的速度   |
| **last_actions**      | 上一时间步的控制动作     | 12   | 历史动作       |
| **command**           | 目标线速度和角速度       | 3    | [vx, vy, vyaw] |

### 台阶地形观察空间（60 维）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (60,), float32)` |
| **维度** | 60                               |

在粗糙地形基础上增加了：

| 部分                   | 内容说明             | 维度 | 备注                   |
| ---------------------- | -------------------- | ---- | ---------------------- |
| **feet_contact_force** | 四个足部的接触力向量 | 12   | 每足 3 维 (Fx, Fy, Fz) |

---

## 奖励函数设计

GO1 复杂地形的奖励函数采用多目标加权设计：

```{literalinclude} ../../../../motrix_envs/src/motrix_envs/locomotion/go1/cfg.py
:language: python
:start-after: '# -- docs-tag-start: go1-reward-config --'
:end-before: '# -- docs-tag-end: go1-reward-config --'
```

_总奖励 = 加权组合以上所有项_

---

## 初始状态

### 粗糙地形初始化

-   **地形生成**：使用高度图生成随机地形
-   **地形高度级别**：预设 -2.5m、0.5m、2.0m 三个高度级别
-   **位置随机化**：在基础训练级别时机器人固定位置，高级别时按预设的 25 个位置周期循环随机选择

### 台阶地形初始化

-   **地形类型**：连续地形上布置多种以台阶为主的小地块
-   **位置随机化**：与粗糙地形类似的位置随机化策略

### 机器人初始化

-   **关节角度**：设置为默认站立姿态，添加 [-0.125, 0.125] 弧度噪声
-   **速度初始化**：所有线速度和角速度初始化为零

## Episode 终止条件

-   **身体接触地面**：机器人躯干与地面发生非预期接触
-   **速度异常**：线速度平方和超过阈值（1e8）

---

## 使用指南

### 1. 环境预览

```bash
uv run scripts/view.py --env go1-rough-terrain-walk
uv run scripts/view.py --env go1-stairs-terrain-walk
```

### 2. 开始训练

```bash
uv run scripts/train.py --env go1-rough-terrain-walk
uv run scripts/train.py --env go1-stairs-terrain-walk
```

### 3. 查看训练进度

```bash
uv run tensorboard --logdir runs/go1-rough-terrain-walk
```

### 4. 测试训练结果

由于粗糙地形场景中同时生成了了一个无限大平面和一个崎岖地形高度场，测试训练结果时会仿照训练过程先将智能体生成在平面上，完成一轮行走后再生成到崎岖地形上。用户需要主动调整相机视角和位置来观察智能体的状态。

```bash
uv run scripts/play.py --env go1-rough-terrain-walk
uv run scripts/play.py --env go1-stairs-terrain-walk
```

---

## 预期训练结果

### 粗糙地形任务 (go1-rough-terrain-walk)

1. 能够适应不同的粗糙地形高度
2. 速度跟踪精度高，姿态稳定
3. 步态协调，足部打滑少

### 台阶地形任务 (go1-stairs-terrain-walk)

1. 能够稳定上下台阶
2. 能够适应不同高度和宽度的台阶
3. 动作流畅，无明显卡顿

## 训练性能参考

### go1-rough-terrain-walk

| 操作系统     | 训练后端 | CPU               | GPU         | 环境数 | 训练时间 (30000 steps) |
| ------------ | -------- | ----------------- | ----------- | ------ | ---------------------- |
| Ubuntu 22.04 | JAX      | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 7m20s                  |
| Ubuntu 22.04 | PyTorch  | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 8m30s                  |
| Windows 11   | PyTorch  | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 10m42s                 |

### go1-stairs-terrain-walk

| 操作系统     | 训练后端 | CPU               | GPU         | 环境数 | 训练时间 (30000 steps) |
| ------------ | -------- | ----------------- | ----------- | ------ | ---------------------- |
| Ubuntu 22.04 | JAX      | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 7m18s                  |
| Ubuntu 22.04 | PyTorch  | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 8m41s                  |
| Windows 11   | PyTorch  | AMD Ryzen 7 9700X | RTX 5070 Ti | 2048   | 10m52s                 |
