# 任务二：Anymal C导航
## 代码位置
1. 环境：.\MotrixLab\motrix_envs\src\motrix_envs\navigation\anymal_c
    * 资产：.\MotrixLab\motrix_envs\src\motrix_envs\navigation\anymal_c\xmls
    * 最优权重：MotrixLab\runs\anymal_c_navigation_flat\25-12-05_11-30-06-232902_PPO\checkpoints\best_agent.pt
2. 配置
   * 修改.\motrix_envs\src\motrix_envs\navigation\__init__.py文件，添加 `anymal_c`目录，即
    ```python
    from . import anymal_c
    ```
   * 修改 .\motrix_rl\src\motrix_rl\cfgs.py, 添加 `anymal_c_navigation_flat` 配置
   即
    ```python
    class navigation:
        @rlcfg("anymal_c_navigation_flat")
        @dataclass
        class AnymalCPPOConfig(PPOCfg):

            # ===== 基础训练参数 =====
            seed: int = 42         # 随机种子
            num_envs: int = 2048               # 训练时并行环境数量
            play_num_envs: int = 16            # 评估时并行环境数量
            max_env_steps: int = 100_000_000   # 最大训练步数
            check_point_interval: int = 100    # 检查点保存间隔（每100次迭代保存一次）

            # ===== PPO算法核心参数 =====
            learning_rate: float = 3e-4        # 学习率
            rollouts: int = 48                 # 经验回放轮数
            learning_epochs: int = 6           # 每次更新的训练轮数
            mini_batches: int = 32             # 小批量数量
            discount_factor: float = 0.99      # 折扣因子
            lambda_param: float = 0.95         # GAE参数
            grad_norm_clip: float = 1.0        # 梯度裁剪

            # ===== PPO裁剪参数 =====
            ratio_clip: float = 0.2            # PPO裁剪比率
            value_clip: float = 0.2            # 价值裁剪
            clip_predicted_values: bool = True # 裁剪预测值

            # 中型网络（默认配置，适合大部分任务）
            policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
            value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    ```

## 训练和测试
1. 训练
    * 运行 .\train.bash 脚本，训练 `anymal_c_navigation_flat` 环境
    * 训练过程中，会在 .\runs 目录下生成训练日志和检查点文件
2. 测试
    * 运行 .\eval.bash 脚本，测试 `anymal_c_navigation_flat` 环境
    * 测试过程中，会在 .\runs 目录下生成测试日志文件

