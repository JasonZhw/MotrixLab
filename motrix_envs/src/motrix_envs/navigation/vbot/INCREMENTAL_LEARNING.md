# MotrixArena 增量学习框架说明

## 核心思想：从Section011到Section012的知识迁移

### 阶段1: Section011（已完成训练）
**任务**：从START(-2.4) → 收集地标 → 2026平台(y=8) → 庆祝
**环境特点**：
- 地形：坑洼地形
- 地标：3笑脸(y=0.1) + 3红包(y=4.1)
- 终点：2026平台 y=8.0
- 训练权重路径：`runs/MotrixArena_S1_section011_56/xxx_PPO_best/checkpoints/best_agent.pt`

### 阶段2: Section012（即将训练）
**任务**：从2026平台(y=8) → 楼梯+吊桥/河床 → 丙午大吉平台(y=24)
**环境特点**：
- 地形：楼梯、吊桥（新地形！）
- 起点：2026平台 [0, 8.0, 2.0]
- 终点：丙午大吉 y=24.0
- **增量学习**：加载section011权重继续训练

---

## 增量学习实施步骤

### Step 1: 准备Section012配置（已完成）

**cfg.py - VBotSection012Cfg**:
```python
@registry.envcfg("MotrixArena_S1_section012_56")
class VBotSection012Cfg(VBotSection01EnvCfg):
    max_episode_seconds = 50.0
    max_episode_steps = 5000
    
    class InitState:
        pos = [0.0, 8.0, 2.0]  # 从2026平台出发
        pos_randomization_range = [-0.2, -0.2, 0.2, 0.2]
    
    class Commands:
        pose_command_range = [-2.0, 24.0, 0.0, 2.0, 24.0, 0.0]  # 目标：丙午大吉
    
    class TaskConfig:
        goal_y = 24.0  # 新终点
        boundary_y_max = 26.0
```

**关键设计**：
- ✅ `InitState.pos = [0, 8.0, 2.0]` - 起点=section011终点
- ✅ `goal_y = 24.0` - 新终点
- ✅ 继承`VBotSection01EnvCfg` - 复用所有奖励函数和终止条件

### Step 2: 加载Section011权重（训练脚本修改）

**方法1: skrl原生增量学习（推荐）**

在 `scripts/train.py` 中添加权重加载：

```python
# 训练section012时指定checkpoint路径
if args.env == "MotrixArena_S1_section012_56":
    pretrained_path = "runs/MotrixArena_S1_section011_56/26-03-02_00-49-20-133608_PPO_best/checkpoints/best_agent.pt"
    
    # skrl加载checkpoint
    agent.load(pretrained_path)
    print(f"✅ 加载section011预训练权重: {pretrained_path}")
```

**方法2: 手动指定checkpoint参数（命令行）**

```bash
uv run scripts/train.py \
  --env MotrixArena_S1_section012_56 \
  --checkpoint runs/MotrixArena_S1_section011_56/.../best_agent.pt
```

### Step 3: 训练策略调整

**Learning Rate降低**（精调已有知识）:
```python
# cfgs.py - Section012PPOConfig
learning_rate = 1e-4  # 从5e-4降低，避免遗忘section011知识
```

**Entropy降低**（减少探索，利用已学策略）:
```python
entropy_loss_scale = 0.005  # 从0.01降低
```

**训练时长缩短**（新地形适应即可）:
```bash
# Section011: 6000 episodes（从零开始）
# Section012: 2000-3000 episodes（微调即可）
```

---

## 增量学习的优势

### 1. **步态迁移**
Section011已学会：
- ✅ 四足行走步态
- ✅ 速度控制（~1.0m/s）
- ✅ 转向控制
- ✅ 倾斜恢复（坑洼地形）

Section012直接复用 → 快速适应楼梯

### 2. **奖励塑形复用**
所有奖励函数继承：
- ✅ 速度跟踪（tracking_speed）
- ✅ 前进投影（forward_progress）
- ✅ 朝向对齐（heading_align）
- ✅ 距离势函数（goal_shaping）
- ✅ 稳定性惩罚（tilt_penalty）

### 3. **观测空间一致**
67维观测完全相同 → 无需重新学习特征提取

---

## 常见问题

### Q1: 硬编码导航会影响已训练权重吗？
**A**: 不会！只要 `use_hardcoded_navigation=False`（默认），权重播放完全按照学到的策略。

### Q2: Section012能否直接用Section011的权重？
**A**: 不能直接用。虽然步态相同，但目标不同（y=8 vs y=24），需要微调。

### Q3: 如何验证迁移学习有效？
**A**: 对比训练曲线：
- 从零训练section012：可能5000+ episodes才收敛
- 增量学习section012：预计2000 episodes即可收敛

### Q4: 如果Section012训练失败怎么办？
**A**: 回退到Section011权重，调整：
1. 降低learning_rate（1e-5）
2. 增加训练时长
3. 检查新地形是否有物理障碍

---

## 下一步计划

1. ✅ **训练Section011完成** - 获得稳定的基础步态
2. 🔄 **准备Section012** - 配置已就绪
3. 🎯 **加载权重训练** - 继承section011知识
4. 📊 **对比评估** - 验证增量学习加速效果
5. 🚀 **Section013** - 继续迁移到最终阶段

---

## 技术细节：权重加载代码示例

```python
# train.py 中添加增量学习支持
import os

def get_pretrained_checkpoint(env_name):
    """自动查找上一阶段的best权重"""
    checkpoint_map = {
        "MotrixArena_S1_section012_56": "runs/MotrixArena_S1_section011_56/*/checkpoints/best_agent.pt",
        "MotrixArena_S1_section013_56": "runs/MotrixArena_S1_section012_56/*/checkpoints/best_agent.pt",
    }
    
    if env_name in checkpoint_map:
        import glob
        matches = glob.glob(checkpoint_map[env_name])
        if matches:
            # 按时间戳排序，取最新
            latest = sorted(matches)[-1]
            return latest
    return None

# 训练主循环前
pretrained = get_pretrained_checkpoint(args.env)
if pretrained and os.path.exists(pretrained):
    agent.load(pretrained)
    print(f"🎯 增量学习：从 {pretrained} 加载权重")
else:
    print("🆕 从零开始训练")
```

---

## 总结

增量学习 = **Section011步态** + **Section012新地形适应**

这样可以：
- ⏱️ 缩短50%训练时间
- 🎯 提高收敛稳定性
- 🧠 保留已学知识
- 🚀 快速适应新任务
