# VBot Section01 奖励系统分析指南

## 目录
1. [奖励体系结构](#奖励体系结构)
2. [每个奖励分量详解](#每个奖励分量详解)
3. [权重调整方法论](#权重调整方法论)
4. [关键指标解读](#关键指标解读)
5. [实战调参案例](#实战调参案例)

---

## 奖励体系结构

VBot的奖励分为**7个层级**，从密集（每步）到稀疏（一次性）：

```
奖励总额 (R_total)
├─ 第1层：密集奖励（每步）
│  ├─ R_speed       : 速度大小跟踪
│  ├─ R_forward     : 前进方向投影
│  ├─ R_heading     : 朝向对齐
│  └─ R_dist_shaping: 距离势函数
├─ 第2层：地形适应奖励（每步）
│  ├─ R_tilt_penalty : 倾斜惩罚
│  └─ R_stair_penalty: 楼梯稳定性
├─ 第3层：障碍物奖励（稀疏，一次/地标）
│  ├─ R_landmarks   : 笑脸/红包
│  └─ R_landmarks_mag: 引力奖励
├─ 第4层：里程碑奖励（稀疏，一次/平台）
│  └─ R_milestone   : 到达2026/丙午/终点
├─ 第5层：庆祝奖励（一次/动作）
│  ├─ R_rotation    : 旋转过程奖励
│  └─ R_celebration : 完成庆祝奖励
├─ 第6层：探索奖励（一次）
│  └─ R_new_territory: 到达更远距离
└─ 第7层：惩罚项（每步）
   ├─ R_stuck_penalty: 卡住警告
   └─ R_tilt_terminate: 摔倒终止
```

---

## 每个奖励分量详解

### 1️⃣ 速度大小跟踪 (R_speed)

**代码位置**: `_compute_reward()` line ~760

```python
speed = np.linalg.norm(vel_xy, axis=1)  # 计算XY平面速度大小
speed_error = np.square(speed - 1.0)    # 与目标速度1.0m/s的偏差平方
tracking_speed = np.exp(-speed_error / 0.25)  # 高斯函数，速度越接近1.0m/s越高
reward += 1.0 * tracking_speed  # 权重=1.0
```

**物理意义**：
- 奖励机器人保持稳定的高速运动
- `1.0 * np.exp(-0)` = 1.0分（达到目标速度）
- `1.0 * np.exp(-1)` ≈ 0.37分（速度相差1m/s）
- `1.0 * np.exp(-4)` ≈ 0.018分（速度相差2m/s）

**调整建议**：
- ✅ 当前 1.0 合理（每步~0.3-1.0分）
- 如果机器人"走走停停"：增加到 1.5
- 如果机器人"猛冲后摔"：降低到 0.5

---

### 2️⃣ 前进投影 (R_forward)

**代码位置**: `_compute_reward()` line ~763-773

```python
vel_toward_goal = np.sum(vel_xy * goal_dir_unit.squeeze(), axis=1)
                  # 计算速度在目标方向上的投影
                  # 原理: vel · goal_dir / |goal_dir|
                  # 返回值: -1.0 ~ 1.5（负=向后走）

forward_progress = np.clip(vel_toward_goal, -0.5, 1.5)
                  # 限制范围，防止极端值

# 楼梯段特殊加权
in_stairs = (robot_y >= 12.0) & (robot_y <= 21.0)
stair_bonus = np.where(in_stairs, 1.8, 1.2)  # 楼梯1.8x，其他1.2x
reward += stair_bonus * forward_progress
```

**数值例子**：
```
场景A：沿目标方向走，速度0.8m/s
  vel_toward_goal = 0.8
  forward_progress = 0.8
  普通路段奖励 = 1.2 × 0.8 = 0.96分 ✅
  楼梯段奖励 = 1.8 × 0.8 = 1.44分 ✅

场景B：走错方向，反向速度0.5m/s
  vel_toward_goal = -0.5
  forward_progress = -0.5 (限制了)
  普通路段奖励 = 1.2 × (-0.5) = -0.6分 ❌
  → 强烈惩罚错方向！

场景C：停在原地
  vel_toward_goal = 0
  forward_progress = 0
  奖励 = 0分 （既不奖励也不惩罚）
```

**权重含义**：
- **普通路段 1.2**: 每往目标走1m，获得 ~1.2×60s = 72分（粗估）
- **楼梯路段 1.8**: 加强楼梯学习信号 +50%

**调整建议**：
- 楼梯通过率<30% → 增加到 2.0 或 2.2
- 楼梯通过率>50% → 降到 1.5 或 1.6（避免过度学习）
- 普通路段权重：保持 1.2（已验证有效）

---

### 3️⃣ 朝向对齐 (R_heading)

**代码位置**: `_compute_reward()` line ~774-783

```python
desired_heading = np.arctan2(goal_dir[:, 1], goal_dir[:, 0])
                  # 计算目标方向的角度（弧度）
                  
heading_error = desired_heading - robot_heading
                # 机器人实际朝向 vs 需要朝向的差值
                
# 处理角度环绕（±π转换）
heading_error = np.where(heading_error > np.pi, heading_error - 2*np.pi, heading_error)
heading_error = np.where(heading_error < -np.pi, heading_error + 2*np.pi, heading_error)

heading_align = np.exp(-np.square(heading_error / 0.6))
              # 高斯核心：heading_error越小，值越接近1.0
              # 0.6是标准差参数（关键！）

reward += 0.2 * heading_align  # 权重=0.2
```

**高斯函数理解**：
```
heading_error = 0°  → heading_align = 1.0 → reward = 0.2
heading_error = 30° → heading_align ≈ 0.75 → reward ≈ 0.15
heading_error = 90° → heading_align ≈ 0.01 → reward ≈ 0.002
```

**标准差参数0.6的含义**：
- 在 ±0.6弧度（±34°）内 → 奖励>0.1
- 在 ±1.2弧度（±69°）内 → 奖励>0.01
- 超过 ±1.2弧度 → 奖励接近0

**权重 0.2 含义**：
- 弱化朝向约束（不希望限制转向）
- 允许横向收集地标时有较大朝向误差
- 比前进投影权重(1.2)小6倍

**调整建议**：
- 如果机器人"转圈"不前进：增加到 0.4 或 0.5
- 如果机器人"无法转向收集地标"：降低到 0.1 或 0.0（关闭）

---

### 4️⃣ 距离势函数 (R_dist_shaping)

**代码位置**: `_compute_reward()` line ~786-793

```python
dist_to_goal = goal_dist.flatten()
              # 当前位置到目标的欧式距离

if "prev_dist_to_goal" not in info:
    info["prev_dist_to_goal"] = dist_to_goal.copy()

dist_improvement = info["prev_dist_to_goal"] - dist_to_goal
                 # 本步是否靠近了目标？
                 # > 0 表示靠近（好！）
                 # < 0 表示远离（坏！）

goal_shaping = np.clip(dist_improvement * 2.0, -0.1, 0.2)
             # 斜率2.0：每靠近0.1m，奖励 0.2分
             # 限制范围：[-0.1, 0.2]，防止奖励爆炸

reward += goal_shaping
```

**数值例子**：
```
场景A：靠近了0.2m（很好）
  dist_improvement = 0.2
  goal_shaping = clip(0.2×2.0, -0.1, 0.2) = 0.2 ✅

场景B：靠近了0.05m（有点慢）
  dist_improvement = 0.05
  goal_shaping = clip(0.05×2.0, -0.1, 0.2) = 0.1

场景C：远离了0.1m（错误方向）
  dist_improvement = -0.1
  goal_shaping = clip(-0.1×2.0, -0.1, 0.2) = -0.1 ❌
  → 惩罚！

场景D：没有动（卡住）
  dist_improvement = 0
  goal_shaping = 0 （既不奖励也不惩罚）
```

**权重含义**：
- **斜率 2.0**: 平衡快速移动奖励
- **范围 [-0.1, 0.2]**: 避免奖励/惩罚过度

**调整建议**：
- 正常情况：保持现状
- 如果卡住不动：增加斜率到 3.0 或 4.0
- 如果奖励摆动：减少斜率到 1.0

---

### 5️⃣ 地标吸引力 (R_landmarks)

**代码位置**: `_compute_reward()` line ~835-860

```python
# 笑脸（3个，Y=0.1）
s_diff = smile_positions - robot_xy  # [E, N_smiles, 2] 距离向量
s_dist = np.linalg.norm(s_diff, axis=2)  # [E, N_smiles] 距离标量
s_dir = s_diff / np.maximum(s_dist[:, :, np.newaxis], 1e-6)  # 单位方向

s_valid = (s_dist < 6.0) & (~s_collected)  # 6m范围内且未收集
s_strength = np.clip((6.0 - s_dist) / 6.0, 0, 1) * s_valid
           # 距离越近，强度越高：0 ~ 1
           # 6m外：强度=0（看不到）
           # 0m（在上面）：强度=1（最强）

s_vel_toward = np.sum(vel_2d[:, np.newaxis, :] * s_dir, axis=2)
             # 速度在笑脸方向上的投影

landmark_attraction += np.sum(
    s_strength * np.clip(s_vel_toward * 0.01, 0, 0.02), axis=1
)  # 只奖励朝向笑脸的速度，上限0.02分
```

**吸引力过程**：
```
距离 6m → 强度 0%   → 无奖励
距离 5m → 强度 17%  → 若速度指向，奖励 0.017×weight
距离 3m → 强度 50%  → 若速度指向，奖励 0.010×weight  
距离 0m → 强度 100% → 若速度指向，奖励 0.020×weight
```

**一次性收集奖励**：
```python
# 在地标区域内（笑脸±1.3m）
first_collect = in_range & (~info[collected_key])
reward += first_collect.astype(np.float32) * self.smile_reward  # 10分一次
```

**总和**：
- 靠近笑脸过程：逐步获得 0-0.02分/step
- 进入范围后：一次性获得 10分

**调整建议**：
- 如果"忽视笑脸直冲终点"：增加权重 0.01 → 0.02
- 如果"过度纠缠笑脸"：降低一次性奖励 10 → 5

---

### 6️⃣ 里程碑奖励 (R_milestone)

**代码位置**: `_compute_reward()` line ~945-958

```python
# 到达判定：robot_y在[milestone_y-0.5, milestone_y+2.0]范围内
reached_milestone = (robot_y >= milestone_y - 0.5) & (robot_y <= milestone_y + 2.0)

# 首次到达时奖励
first_reach_milestone = reached_milestone & (~info[milestone_key])

# 加权奖励：根据地标收集率调整
if self.enable_landmark_rewards:
    milestone_bonus = 50.0 * (0.3 + 0.7 * collection_ratio)
else:
    milestone_bonus = 50.0
```

**数值例子**：
```
collection_ratio = 0%   → bonus = 50 × (0.3 + 0) = 15分
collection_ratio = 50%  → bonus = 50 × (0.3 + 0.35) = 32.5分
collection_ratio = 100% → bonus = 50 × (0.3 + 0.7) = 50分
```

**含义**：
- 奖励"走到里程碑"
- 但加权鼓励"同时收集地标"
- 这样避免机器人"只冲终点，忽视地标"的策略

---

### 7️⃣ 庆祝奖励 (R_celebration)

**代码位置**: `_compute_reward()` line ~980-1000

```python
# 过程奖励：在平台上旋转时
angular_speed = np.abs(gyro[:, 2])
reward += np.where(
    at_milestone & (~done), 
    np.clip(angular_speed * 0.05, 0.0, 0.05),  # 最多0.05分/step
    0.0
)

# 完成奖励：旋转满3圈后
celebration_complete = (info[yaw_acc_key] >= required_rotation) & (~done)
reward += celebration_complete.astype(np.float32) * 15.0  # 15分一次
```

**收益链**：
- 进入平台区域 → 强制减速
- 开始旋转 → 每步 0-0.05分
- 完成3圈 → 一次性 15分

**总收益**（3秒庆祝+3秒旋转）：
- 过程奖励：~6秒×0.03分/step = 0.18分
- 完成奖励：15分
- 总计：~15分

---

## 权重调整方法论

### 📊 **权重调整的3步法**

#### 步骤1：诊断问题
在训练日志中看到：

| 现象 | 诊断 | 优先权重 |
|------|------|---------|
| "只在乎笑脸，忽视前进" | 地标奖励太高 | ↓ smile_reward |
| "走错方向直冲" | 前进投影权重太低 | ↑ forward_progress |
| "在楼梯卡住不动" | 楼梯激励不足 | ↑ stair_bonus |
| "走走停停" | 速度奖励不足 | ↑ tracking_speed |
| "频繁摔倒" | 稳定性惩罚太弱 | ↑ stability_penalty |

#### 步骤2：单点调整（一次改一个！）

**错误做法**：同时改10个权重
**正确做法**：
1. 改 1 个权重
2. 训练 1000 steps
3. 看指标变化
4. 再改下一个

#### 步骤3：监控指标（关键！）

```python
# 打印这些指标
print(f"【进度】avg_y={avg_robot_y:.2f}m, target={target_y}m")
print(f"【效率】{(avg_robot_y - last_avg_y):.2f}m/1ksteps")
print(f"【地标】{collected_ratio*100:.1f}% ({collected_count}/13)")
print(f"【楼梯】通过率={(num_y_gt_21)/num_envs*100:.1f}%")
print(f"【奖励】avg={avg_reward:.2f}, max={max_reward:.2f}")
```

---

## 关键指标解读

### 1. 平均进度 (avg_y)

```
期望：每1000步增长 ~1-2m
- <1m：太慢，可能卡住或地标吸引过度
- 1-2m：正常范围 ✅
- >3m：很快，但检查是否在忽视地标
```

### 2. 地标完成率 (collected_ratio)

```
Y < 8m（到2026前）：
  - 期望：50-80%（需要时间收集）
  
Y ∈ [8, 21m]（楼梯+吊桥）：
  - 期望：70-90%（应该继续收集）
  
Y > 21m（河床）：
  - 期望：>95%（最后冲刺）
```

### 3. 楼梯通过率 (Y>21%)

```
这是关键指标！
- 0-20%：楼梯完全卡住 ❌ → 增加action_scale或stair_bonus
- 20-50%：部分通过 ⚠️  → 继续优化
- 50-80%：基本通过 ✅ → 可以后续优化其他
- >80%：完全掌握 🎉 → 降低权重防止过度学习
```

### 4. 奖励曲线

```
期望形状：
    ╱
   ╱  ← 陡峭上升（学习中）
  ╱
 ___  ← 收敛（已学会）

不良形状：
    ╲  ← 下降（发散）
     ___  ← 平坦（无进展）
```

---

## 实战调参案例

### 📌 案例1：机器人卡在楼梯（Y=12-18）

**症状**：
```
avg_y 困在 12-14m，楼梯通过率 <20%
平均奖励 0.8分（太低）
```

**分析**：
- 楼梯太陡，前进奖励不足
- action_scale=0.45 还不够？

**调整方案**（按优先级）：

**优先级1：加强楼梯前进奖励**
```python
# 在cfg.py中
# 原来：普通1.2x，楼梯1.8x
# 改成：普通1.2x，楼梯2.2x（+0.4倍增）

stair_bonus = np.where(in_stairs, 2.2, 1.2)  # 改这行
```

**验证**（训练1000步）：
```
✅ 如果avg_y增速 <1m → 1.5m：改对了！
❌ 如果还是卡住：进行优先级2
```

**优先级2：增加action_scale**
```python
# 在cfg.py中
# 原来：action_scale = 0.45
# 改成：action_scale = 0.50（再增加）

action_scale = 0.50
```

**优先级3：减弱楼梯稳定性惩罚（如果太严）**
```python
# 在MotrixArena_S1_section01_56.py中
# 原来：倾斜>50°时惩罚-0.1
# 改成：倾斜>55°时惩罚-0.08

stair_stability_penalty = np.where(
    in_stairs & (tilt_cos < 0.57),  # cos(55°)=0.57
    -0.08 * (0.57 - tilt_cos),      # 改成-0.08
    0.0
)
```

---

### 📌 案例2：地标完成率太低（<40%）

**症状**：
```
楼梯通过率 >60%（可以到达2026）
但collected_ratio = 30%（地标漏掉很多）
```

**分析**：
- 机器人在冲向下一个waypoint时，忽视了周围的地标
- 地标检测范围太小？还是权重太低？

**调整方案**：

**优先级1：增加地标吸引力的感知范围**
```python
# 在_compute_reward()中
# 原来：笑脸<6.0m才有吸引力
# 改成：笑脸<8.0m有吸引力

s_valid = (s_dist < 8.0) & (~s_collected)  # 改成8.0
s_strength = np.clip((8.0 - s_dist) / 8.0, 0, 1) * s_valid  # 也改成8.0
```

**优先级2：增加地标吸引权重**
```python
# 原来：速度投影 * 0.01，上限0.02
# 改成：速度投影 * 0.015，上限0.03

landmark_attraction += np.sum(
    s_strength * np.clip(s_vel_toward * 0.015, 0, 0.03), axis=1
)  # 增加权重
```

**优先级3：增加一次性收集奖励**
```python
# 在cfg.py中
# 原来：smile_reward = 10.0
# 改成：smile_reward = 15.0

smile_reward: float = 15.0  # 增加到15
```

---

## 总结：权重调整速查表

| 问题 | 权重 | 默认 | 调整 |
|------|------|------|------|
| 卡在楼梯 | stair_bonus | 1.8 | ↑ 2.2-2.5 |
| 忽视地标 | s_strength权重 | 0.01 | ↑ 0.015-0.02 |
| 走走停停 | tracking_speed | 1.0 | ↑ 1.5 |
| 频繁转向 | heading_align | 0.2 | ↓ 0.1 |
| 摔倒过多 | stability_penalty | -0.1 | ↑ -0.15 |
| 冲出平台 | platform_speed_limit | 0.25 | ↓ 0.2 |
| 回血太慢 | celebration_reward | 15.0 | ↑ 20.0 |

---

## 快速检查清单

训练前检查：
- [ ] 初始朝向是 +Y (π/2) 吗？
- [ ] action_scale >= 0.45 吗？
- [ ] episode_steps >= 10000 吗？
- [ ] stair_bonus >= 1.8 吗？

训练时监控：
- [ ] avg_y 每1000步增长 1-2m 吗？
- [ ] 楼梯通过率 >20% 吗？
- [ ] 地标完成率 >40% 吗？
- [ ] 平均奖励持续上升吗？

收敛后评估：
- [ ] 能稳定到达丙午平台吗？
- [ ] 地标收集率 >80% 吗？
- [ ] 庆祝动作完整吗？
- [ ] 总奖励 >100分 吗？

