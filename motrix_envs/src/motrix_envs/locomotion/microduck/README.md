# Microduck 平地行走训练

这个任务使用 PPO 和 RSL-RL，让
[Microduck](https://github.com/pollen-robotics/microduck)
在 MotrixLab 仿真中学习站立、向前走、侧向移动、转弯和停止。

环境名称：

```text
microduck-flat-terrain-walk
```

下面的命令都需要在 MotrixLab 仓库根目录执行。

> 当前策略只用于仿真。没有经过执行器建模、域随机化和真机安全验证前，
> 不要直接部署到实体 Microduck。

## 1. 安装 RSL-RL 依赖

已经成功训练过其他 RSL-RL 任务时可以跳过这一步。

```bash
uv sync --all-packages --extra rslrl
```

## 2. 从头开始训练

```bash
uv run scripts/train.py \
  --env=microduck-flat-terrain-walk \
  --sim-backend=np \
  --rllib=rslrl \
  --train-backend=torch \
  --num-envs=512 \
  --seed=42
```

- `--num-envs=512` 是比较稳妥的起点。显存充足时可以提高到 1024、2048 或 4096。
- 默认最多训练 5,000 个 PPO iteration，配置位于
  `motrix_rl/src/motrix_rl/tasks/microduck.py`。
- 5,000 只是训练上限，不代表必须全部跑完。当前实验在约 1,300 轮已经表现良好，
  可以先播放 checkpoint，再决定是否继续等待。
- RSL-RL 每 50 轮保存一次 `model_<iteration>.pt`。如果使用 `Ctrl+C` 提前停止，
  中断瞬间不会额外保存，但之前每 50 轮保存的模型仍然存在。
- 当前原版训练入口不支持 checkpoint 续训。重新执行训练命令会创建一个新实验，并从第 0 轮开始。

默认配置已经包含完整行走训练所需的命令采样和难度设置，不需要手动切换训练阶段，
直接使用上面的命令训练即可。强化学习存在随机性，因此仍需通过 TensorBoard 和播放 checkpoint
确认最终效果。

训练结果保存在：

```text
runs/microduck-flat-terrain-walk/rslrl/<实验目录>/
```

## 3. 用 TensorBoard 看训练过程

新开一个终端：

```bash
uv run tensorboard \
  --logdir=runs/microduck-flat-terrain-walk/rslrl \
  --host=127.0.0.1 \
  --port=6006
```

浏览器打开 <http://127.0.0.1:6006>。

建议重点观察下面这些曲线：

- `Train/mean_reward`：总成绩。总体上升说明策略正在学习，短期波动是正常现象。
- `Train/mean_episode_length`：每回合平均存活步数。越长通常说明越不容易摔倒。
- `Episode/metrics/linear_velocity_error`：实际速度和目标速度的误差，越低越好。
- `Episode/metrics/single_support_fraction`：只有一只脚着地的时间比例。
  从接近 0 变成稳定的非零值，通常说明小鸭子开始交替抬脚，而不是双脚拖行；
  这个值不是越接近 1 越好。
- `Episode/Reward/termination`：摔倒产生的惩罚。它是负值，越接近 0 越好。

不要只根据 `mean_reward` 判断模型。最终还需要播放策略，亲眼检查是否会摔倒、
是否真的迈步，以及能否按照命令改变速度和方向。

## 4. 播放并检查训练策略

“策略”就是训练得到的神经网络，也就是实验目录中的 `model_*.pt` 文件。

### 自动播放最新 checkpoint

```bash
uv run scripts/play.py \
  --env=microduck-flat-terrain-walk \
  --sim-backend=np \
  --rllib=rslrl \
  --num-envs=1
```

没有指定 `--policy` 时，播放脚本会从该环境最近一次训练实验中自动选择最新的 checkpoint。
如果同时存在 RSL-RL 和 SKRL 实验，建议用下面的方法明确指定文件。

### 播放指定 checkpoint

先查看实验目录里的模型：

```bash
ls runs/microduck-flat-terrain-walk/rslrl/*/model_*.pt
```

然后指定其中一个文件，例如：

```bash
uv run scripts/play.py \
  --env=microduck-flat-terrain-walk \
  --sim-backend=np \
  --rllib=rslrl \
  --num-envs=1 \
  --policy=runs/microduck-flat-terrain-walk/rslrl/<实验目录>/model_1300.pt
```

修改文件名即可对比不同训练轮数的 checkpoint，直观看到站立、迈步和速度跟随是否持续改善。

### 训练效果

一个比较好的策略应该满足：

- 能连续站立和行走，不频繁摔倒或趴下；
- 左右脚能够交替迈步，而不是靠滑动或双脚一起跳；
- 实际前进速度能跟随目标速度；
- 能随机性的稳定站住；
- 在不同运动命令下，能完成侧移、转弯和少量后退；
- 关节动作没有高频抖动，也不会频繁撞到关节极限。

如果 1,300 轮模型已经满足这些条件，就可以把它作为当前候选策略。
继续训练到 5,000 轮的意义是观察步态还能否更平滑、更准确，而不是单纯追求更大的轮数。
