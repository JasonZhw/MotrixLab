# Unitree GO1 平地行走

Unitree GO1 是一个四足机器人平台，本示例展示了如何训练 GO1 在平坦地形上实现稳定的步态行走。

```{video} /_static/videos/go1_walk.mp4
:poster: _static/images/poster/go1_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

GO1 四足机器人具有 12 个自由度（每条腿 3 个关节），需要通过深度强化学习学习协调的步态控制。该环境使用 MotrixSim 物理引擎进行仿真，提供高保真的动力学模拟。智能体通过控制各关节的目标位置（通过 PD 控制器转换为力矩）来实现速度跟踪和姿态稳定。

---

## 动作空间（Action Space）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-1.0, 1.0, (12,), float32)` |
| **维度** | 12                               |

动作对应 12 个关节的位置控制指令（相对于默认站立姿态的偏移量），包括：

| 序号 | 动作含义（关节位置变化）   | 对应肢体 |
| ---: | -------------------------- | :------: |
|  0-2 | 髋关节、大腿关节、小腿关节 |  前左腿  |
|  3-5 | 髋关节、大腿关节、小腿关节 |  后左腿  |
|  6-8 | 髋关节、大腿关节、小腿关节 |  前右腿  |
| 9-11 | 髋关节、大腿关节、小腿关节 |  后右腿  |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (48,), float32)` |
| **维度** | 48                               |

GO1 环境的观测空间由以下部分组成（按顺序）：

| 部分                  | 内容说明         | 维度 | 备注               |
| --------------------- | ---------------- | ---- | ------------------ |
| **noisy_linvel**      | 局部坐标系线速度 | 3    | 带噪声的线速度     |
| **noisy_gyro**        | 陀螺仪数据       | 3    | 带噪声的角速度     |
| **local_gravity**     | 局部重力方向     | 3    | 重力向量投影       |
| **noisy_joint_angle** | 关节角度         | 12   | 相对于默认值的偏差 |
| **noisy_joint_vel**   | 关节速度         | 12   | 带噪声的关节速度   |
| **last_actions**      | 上一帧动作       | 12   | 历史动作信息       |
| **command**           | 速度命令         | 3    | [vx, vy, vyaw]     |

---

## 奖励函数设计

GO1 的奖励函数是一个复杂的复合函数，包含多个组件：

```{literalinclude} ../../../../motrix_envs/src/motrix_envs/locomotion/go1/cfg.py
:language: python
:start-after: '# -- docs-tag-start: go1-reward-config --'
:end-before: '# -- docs-tag-end: go1-reward-config --'
```

_总奖励 = 加权组合以上所有项_

---

## 初始状态

-   **机器人位置**：固定在初始位置
-   **关节角度**：设置为默认站立姿态
-   **关节角度噪声**：每个关节在 [-0.125, 0.125] 弧度范围内添加随机噪声
-   **速度初始化**：所有线速度和角速度初始化为零

## Episode 终止条件

-   **身体接触地面**：机器人躯干与地面发生非预期接触
-   **速度异常**：线速度平方和超过阈值（1e8）

---

## 使用指南

### 1. 环境预览

```bash
uv run scripts/view.py --env go1-flat-terrain-walk
```

### 2. 开始训练

```bash
uv run scripts/train.py --env go1-flat-terrain-walk
```

### 3. 查看训练进度

```bash
uv run tensorboard --logdir runs/go1-flat-terrain-walk
```

### 4. 测试训练结果

```bash
uv run scripts/play.py --env go1-flat-terrain-walk
```

---

## 预期训练结果

1. 稳定的四足步态（trot 步态或其他协调步态）
2. 良好的速度跟踪能力
3. 能够跟踪不同的速度命令（前进、转向）
4. 姿态稳定，无明显侧翻
