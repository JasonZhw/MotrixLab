# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import (
    RslrlCfg,
)
from motrix_rl.skrl.config import (
    SkrlCfg,
)


# docs-start: cartpole-skrl-config
@rlcfg("cartpole")
@dataclass
class CartPoleSkrlPpo(SkrlCfg):
    """CartPole SKRL configuration with complete explicit parameter filling.

    All parameters from parent classes are explicitly specified.
    """

    def __post_init__(self):
        """Configure SKRL runner settings with explicit parameters."""
        # Environment settings
        self.num_envs = 2048  # Number of parallel environments during training
        self.play_num_envs = 16  # Number of parallel environments during evaluation

        # Get runner and nested configs
        runner = self.runner
        models = runner.models
        agent = runner.agent
        trainer = runner.trainer

        # Random seed
        runner.seed = 42  # Random seed for reproducibility

        # Models configuration
        models.separate = False  # Share features between policy and value networks

        # Policy network configuration
        models.policy.class_name = "GaussianMixin"  # Use Gaussian policy for continuous actions
        models.policy.clip_actions = False  # Don't clip actions to action space
        models.policy.clip_log_std = True  # Clip log standard deviation
        models.policy.initial_log_std = 1.0  # Initial log standard deviation
        models.policy.min_log_std = -20.0  # Minimum log standard deviation
        models.policy.max_log_std = 2.0  # Maximum log standard deviation
        models.policy.reduction = "sum"  # Reduction method for loss computation
        models.policy.input = "STATES"  # Input to policy network
        models.policy.hiddens = [32, 32]  # Hidden layer sizes (small network for simple task)
        models.policy.hidden_activation = ["elu"]  # Activation function for hidden layers
        models.policy.output = "ACTIONS"  # Output of policy network
        models.policy.output_activation = ""  # No activation for output layer
        models.policy.output_scale = 1.0  # Scale factor for output

        # Value network configuration
        models.value.class_name = "DeterministicMixin"  # Use deterministic value function
        models.value.clip_actions = False  # Don't clip actions
        models.value.input = "STATES"  # Input to value network
        models.value.hiddens = [32, 32]  # Hidden layer sizes (small network for simple task)
        models.value.hidden_activation = ["elu"]  # Activation function for hidden layers
        models.value.output = "ONE"  # Output single value
        models.value.output_activation = ""  # No activation for output layer
        models.value.output_scale = 1.0  # Scale factor for output

        # Memory configuration
        runner.memory.class_name = "RandomMemory"  # Use random sampling memory
        runner.memory.memory_size = -1  # Unlimited memory size (-1 means auto-calculate)

        # Agent configuration
        agent.class_name = "PPO"  # Use Proximal Policy Optimization algorithm
        agent.rollouts = 32  # Number of experience rollouts to collect
        agent.learning_epochs = 5  # Number of learning epochs per update (higher than default 2)
        agent.mini_batches = 4  # Number of mini-batches (fewer than default 32 for simple task)
        agent.discount_factor = 0.99  # Discount factor (gamma) for future rewards
        agent.lam = 0.95  # GAE (Generalized Advantage Estimation) lambda parameter
        agent.learning_rate = 1e-3  # Learning rate for optimizer
        agent.learning_rate_scheduler = "KLAdaptiveLR"  # Use KL-divergence adaptive learning rate
        agent.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}  # KL threshold for adaptive LR
        agent.random_timesteps = 0  # Number of random timesteps before using policy
        agent.learning_starts = 0  # Timesteps before learning starts
        agent.grad_norm_clip = 1.0  # Maximum gradient norm for clipping
        agent.ratio_clip = 0.2  # PPO clipping ratio for policy update
        agent.value_clip = 0.2  # Clipping parameter for value function loss
        agent.clip_predicted_values = True  # Clip predicted values in value loss
        agent.entropy_loss_scale = 0.0  # Coefficient for entropy loss (disabled)
        agent.value_loss_scale = 2.0  # Coefficient for value function loss
        agent.kl_threshold = 0  # KL divergence threshold (0 means disabled)
        agent.rewards_shaper_scale = 1.0  # Scale factor for reward shaping
        agent.time_limit_bootstrap = True  # Use bootstrapping for time-limited episodes

        # Experiment configuration
        agent.experiment.directory = "runs"  # Directory to save experiment results
        agent.experiment.experiment_name = ""  # Experiment name (empty means auto-generated)
        agent.experiment.write_interval = -1  # TensorBoard write interval (-1 means default)
        agent.experiment.checkpoint_interval = -1  # Checkpoint save interval (-1 means default)

        # Trainer configuration
        trainer.class_name = "SequentialTrainer"  # Use sequential trainer
        trainer.timesteps = 5000  # Total training timesteps (sufficient for CartPole)


# docs-end: cartpole-skrl-config


# docs-start: cartpole-rslrl-config
@rlcfg("cartpole")
@dataclass
class CartPoleRslrlPpo(RslrlCfg):
    """CartPole RSLRL configuration with complete explicit parameter filling.

    All parameters from parent classes are explicitly specified.
    """

    def __post_init__(self):
        """Configure RSLRL runner settings with explicit parameters."""
        # Environment settings
        self.num_envs = 2048  # Number of parallel environments during training
        self.play_num_envs = 16  # Number of parallel environments during evaluation

        # Get runner and nested configs
        runner = self.runner
        actor = runner.actor
        critic = runner.critic
        algo = runner.algorithm

        # Runner settings
        runner.class_name = "OnPolicyRunner"  # Use on-policy runner
        runner.seed = 42  # Random seed for reproducibility
        runner.device = "cuda:0"  # Device to use for training
        runner.num_steps_per_env = 16  # Number of steps to collect per environment
        runner.max_iterations = 300  # Total number of training iterations
        runner.save_interval = 50  # Checkpoint save interval
        runner.experiment_name = "cartpole"  # Experiment name for logging
        runner.run_name = ""  # Run name (empty means auto-generated)
        runner.logger = "tensorboard"  # Logger type
        runner.obs_groups = {"actor": ["policy"], "critic": ["policy"]}  # Observation groups

        # Actor network configuration
        actor.class_name = "MLPModel"  # Use MLP model
        actor.hidden_dims = [32, 32]  # Hidden layer sizes (small network for simple task)
        actor.activation = "elu"  # Activation function for hidden layers
        actor.obs_normalization = True  # Normalize observations
        actor.stochastic = True  # Use stochastic policy
        actor.init_noise_std = 1.0  # Initial noise standard deviation
        actor.noise_std_type = "scalar"  # Noise std type (scalar or log)
        actor.state_dependent_std = False  # Use state-dependent std

        # Critic network configuration
        critic.class_name = "MLPModel"  # Use MLP model
        critic.hidden_dims = [32, 32]  # Hidden layer sizes (small network for simple task)
        critic.activation = "elu"  # Activation function for hidden layers
        critic.obs_normalization = True  # Normalize observations
        critic.stochastic = False  # Use deterministic value function

        # PPO algorithm configuration
        algo.class_name = "PPO"  # Use PPO algorithm
        algo.optimizer = "adam"  # Optimizer type
        algo.learning_rate = 5.0e-4  # Learning rate for optimizer
        algo.num_learning_epochs = 2  # Number of learning epochs per iteration
        algo.num_mini_batches = 4  # Number of mini-batches for optimization
        algo.schedule = "adaptive"  # Learning rate schedule
        algo.value_loss_coef = 1.0  # Value loss coefficient
        algo.clip_param = 0.2  # PPO clipping parameter
        algo.use_clipped_value_loss = True  # Use clipped value loss
        algo.desired_kl = 0.008  # Desired KL divergence for adaptive learning
        algo.entropy_coef = 5e-3  # Entropy coefficient for exploration
        algo.gamma = 0.99  # Discount factor
        algo.lam = 0.95  # GAE lambda parameter
        algo.max_grad_norm = 1.0  # Maximum gradient norm for clipping
        algo.normalize_advantage_per_mini_batch = False  # Normalize advantage per mini-batch
        algo.rnd_cfg = None  # RND configuration (disabled)
        algo.symmetry_cfg = None  # Symmetry configuration (disabled)


# docs-end: cartpole-rslrl-config
