# Section01 任务要求文档（给 AI 提示词）

## 1. 任务背景
你在维护 MotrixLab 的四足机器人导航任务：`MotrixArena_S1_section01_56`。
当前采用**硬编码 waypoint 导航**，核心目标是：稳定通过坑洼、楼梯、吊桥、河床、滚球，最终到达终点并完成平台庆祝动作。

参考来源：
- 环境实现：`motrix_envs/src/motrix_envs/navigation/vbot/MotrixArena_S1_section01_56.py`
- 环境配置：`motrix_envs/src/motrix_envs/navigation/vbot/cfg.py`
- 训练配置：`motrix_rl/src/motrix_rl/cfgs.py`
- 历史说明：`FULL_ROUTE_TRAINING.md`、`REWARD_ANALYSIS_GUIDE.md`、`REWARD_SYSTEM_V7_SUMMARY.md`

---

## 2. 事实基线（必须遵守）
1. **waypoint 总数是 24 个（索引 0~23）**，
2. milestone waypoint 索引：`[6, 14, 23]`。
3. 当前诊断指标已包含：
   - `avg_y`
   - `avg_reward`
   - `stair_pass`
   - `stuck_ratio`
   - `max_waypoint`
4. `landmark_ratio` 指标已移除，不应再添加回去。
5. 奖励系统已移除 landmark 吸引/收集逻辑，主导航依赖 waypoint + 进度/稳定性奖励。
6. Section01 当前控制幅度：`action_scale = 0.50`（已调优）。
7. Section012（楼梯阶段）控制幅度：`action_scale = 0.45`（已调优）。
8. Section01 PPO 当前关键参数：
   - `learning_rate = 1.5e-4`
   - `discount_factor = 0.995`
   - `lambda_param = 0.97`
   - `entropy_loss_scale = 0.008`

---

## 3. AI 修改代码时的硬约束
1. **最小改动原则**：只改任务相关文件，不做无关重构。
2. 必须保留现有 public 接口与环境注册名。
3. 不得恢复已删除的 landmark 吸引/收集奖励。
4. 不得把 waypoint 数量写回 25 或 30。
5. 修改后需保证：
   - Python 语法通过
   - 指标打印逻辑可运行
   - 不引入新的 KeyError/NameError

---

## 4. 典型任务类型（给 AI 的工作范围）
- 调整楼梯通过稳定性（奖励权重、惩罚阈值、课程出生策略）
- 优化 waypoint 进度相关指标
- 调整 PPO 超参数（学习率、熵、折扣等）
- 修复训练/推理阶段崩溃问题
- 提升平台庆祝动作完成率

---

## 5. 验收标准（每次改动都要满足）
1. 代码语法正确。
2. 每 1000 步诊断输出正常。
3. 输出中包含 `max_waypoint`，不包含 `landmark_ratio`。
4. 训练能启动，不因新增字段/键名导致报错。
5. 关键行为目标：楼梯段（Y=12~21）通过率上升或至少不退化。

---

## 6. 建议 AI 输出格式
1. 先给“改动摘要”（3~6条）。
2. 列出改动文件路径。
3. 说明为什么这样改（对应楼梯/稳定性/探索目标）。
4. 给出验证结果（语法/运行）。
5. 若有风险，单列“风险与回滚点”。

---

## 7. 可直接粘贴给 AI 的提示词模板
你现在是本项目的强化学习环境优化工程师。请基于以下约束修改代码：
- 任务环境：MotrixArena_S1_section01_56
- waypoint 总数固定 24（索引 0~23），milestone 索引 [6,14,23]
- 保留硬编码 waypoint 导航
- landmark_ratio 不允许恢复
- 不允许恢复 landmark 吸引/收集奖励
- 优先优化楼梯段通过率与稳定性（Y=12~21）
- 修改要最小化，避免无关重构
- 改完后给出：改动摘要、文件列表、参数变化、验证结论

当前关键参数基线：
- Section01 action_scale=0.50
- Section01 PPO: lr=1.5e-4, gamma=0.995, lambda=0.97, entropy=0.008

请在不破坏现有训练流程的前提下完成优化。
