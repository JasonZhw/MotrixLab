# Microduck 强化学习任务

这里提供 3 个使用 PPO 和 RSL-RL 的 Microduck 仿真任务：

| 环境名称 | 学习内容 | 默认训练上限 |
| --- | --- | ---: |
| `microduck-flat-terrain-walk` | 站立、行走、侧移、转弯和停止 | 5,000 iterations |
| `microduck-flat-terrain-rollers` | 穿轮滑鞋原地逆时针转一圈，再刹停站稳 | 8,000 iterations |
| `microduck-ground-pick` | 低头让嘴尖接近地面，再稳定恢复站立 | 5,000 iterations |

模型和动作设计参考了
[Microduck](https://github.com/pollen-robotics/microduck) 与
[microduck_rl](https://github.com/pollen-robotics/microduck_rl)。三个策略都保持相同的
61 维观测和 14 维舵机动作，便于后续接入统一的运行时。

> 当前策略只用于仿真。没有经过执行器建模、域随机化和真机安全验证前，
> 不要直接部署到实体 Microduck。

下面的命令都需要在 MotrixLab 仓库根目录执行。

## 1. 安装 RSL-RL 依赖

已经成功训练过其他 RSL-RL 任务时可以跳过这一步。

```bash
uv sync --all-packages --extra rslrl
```

## 2. 从头开始训练

### 平地行走

```bash
uv run scripts/train.py \
  --env=microduck-flat-terrain-walk \
  --sim-backend=np \
  --rllib=rslrl \
  --train-backend=torch \
  --num-envs=512 \
  --seed=42
```

### 轮滑

```bash
uv run scripts/train.py \
  --env=microduck-flat-terrain-rollers \
  --sim-backend=np \
  --rllib=rslrl \
  --train-backend=torch \
  --num-envs=1024 \
  --seed=42
```

### 低头拾取

```bash
uv run scripts/train.py \
  --env=microduck-ground-pick \
  --sim-backend=np \
  --rllib=rslrl \
  --train-backend=torch \
  --num-envs=512 \
  --seed=42
```


## 3. 用 TensorBoard 看训练过程

新开一个终端，把 `<环境名称>` 换成上表中的名称：

```bash
uv run tensorboard \
  --logdir=runs/<环境名称>/rslrl \
  --host=127.0.0.1 \
  --port=6006
```

浏览器打开 <http://127.0.0.1:6006>。


## 4. 播放并检查策略

不指定 `--policy` 时，脚本会自动选择该环境最新实验中的最新 checkpoint：

```bash
uv run scripts/play.py \
  --env=<环境名称> \
  --sim-backend=np \
  --rllib=rslrl \
  --num-envs=1
```

播放指定 checkpoint：

```bash
uv run scripts/play.py \
  --env=<环境名称> \
  --sim-backend=np \
  --rllib=rslrl \
  --num-envs=1 \
  --policy=runs/<环境名称>/rslrl/<实验目录>/model_1300.pt
```
