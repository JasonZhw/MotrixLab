# Basic Framework

MotrixLab is a robot reinforcement learning platform. This section introduces MotrixLab's framework design and the relationships between various components. If you are already familiar with reinforcement learning content, you can skip directly to the next section to learn how to develop your own training environments.

## MotrixLab's Framework Design

MotrixLab adopts a layered architecture design, clearly separating training environments from training logic:

```
MotrixLab/
├── motrix_envs/               # Environment layer: Physics simulation and task definition
│   ├── basic/                  # Basic environments (cartpole, walker, etc.)
│   ├── locomotion/             # Locomotion environments (GO1 robot, etc.)
│   ├── np/                     # NumPy simulation backend framework
│   ├── base.py                 # Environment base class
│   └── registry.py             # Environment registry system
├── motrix_rl/                # Training layer: RL algorithms and configuration
│   ├── skrl/                   # SKRL framework integration (JAX/PyTorch)
│   ├── rslrl/                  # RSLRL framework integration (PyTorch)
│   ├── base.py                 # RL configuration base class
│   └── registry.py             # RL configuration registry system
└── scripts
    ├── train.py                # Training entry script
    ├── play.py                 # Testing entry script
    └── view.py                 # Visualization script
```

## Core Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│                    train.py │ play.py │ view.py                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│             Training Algorithm Layer (SKRL / RSLRL)             │
│                 PPO Trainer │ Network Architecture │ Optimizer   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Environment Implementation Layer            │
│   Environment Config(EnvCfg) │ Environment Impl(Env) │ Reward   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Physics Simulation Layer (MotrixSim)          │
│                    MJCF Model │ Physics Engine │ Collision      │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Core Components

### 1. Training Environment

**Location**: Environment Implementation Layer

The training environment is the core component of MotrixLab, containing three key parts:

-   **Environment Configuration (EnvCfg)**: Defines physics simulation parameters (model files, time steps, episode length, etc.) and task-specific parameters
-   **Environment Implementation (Env)**: Inherits from base environment class, implements specific task logic, physics simulation interaction, and termination condition checking
-   **Reward Function (Reward)**: Implemented in the environment's step method, calculates reward values based on current state and actions

Environments are registered to the system through decorators.

### 2. Reward Function

**Location**: Configuration Management Layer + Environment Implementation Layer

The reward function in MotrixLab adopts a dual-structure design:

-   **Configuration Level**: Define reward weights, reward component types, and scaling parameters in configuration classes
-   **Implementation Level**: Calculate specific reward values in the environment's `_compute_reward` method based on configuration parameters

This design allows reward functions to be flexibly adjusted through configuration files while implementing complex computational logic in code.

### 3. Configuration Parameters

**Location**: Configuration Management Layer

Configuration parameters adopt a hierarchical management structure:

-   **Environment Configuration (EnvCfg)**: Controls physics simulation and task behavior, including simulation parameters, reset noise, time limits, etc.
-   **Training Configuration (RLCfg)**: Controls reinforcement learning algorithms, including network structure, learning rate, batch size, training steps, etc.

Configuration classes support inheritance, parameter validation, and runtime overriding, ensuring parameter reasonableness and flexibility.

### 4. Registry System

**Location**: Hub connecting various components

The registry system implements automatic component registration through the decorator pattern:

-   Environment configuration classes are registered through `@registry.envcfg()`
-   Environment implementation classes are registered through `@registry.env()`, supporting multiple backends
-   RL configuration classes are registered through `@registry.rlcfg()`

The registry system achieves component decoupling, making it simple and fast to add new environments or modify configurations.

## Data Flow and Workflow

### Training Process Overview

```
User Command → Configuration Parsing → Environment Creation → Training Loop → Model Save
   ↓
train.py --env cartpole
   ↓
Find Configuration Classes → Create Environment → Start PPO Training → Save Model
```

### Core Workflow

1. **Environment Definition**: Create environment configuration classes and implementation classes in `src/motrix_envs/`
2. **Automatic Registration**: Register components to the system through decorators
3. **Configuration Loading**: When starting from command line, the system automatically finds and loads corresponding configurations
4. **Environment Creation**: Factory pattern creates environment instances, supporting parameter override
5. **Training Execution**: PPO algorithm interacts with the environment, collects data and updates policy
6. **Result Saving**: Periodically save checkpoints and final models

### Role of Configuration Parameters

Configuration parameters play a key connecting role throughout the process:

-   **Environment Configuration** determines physics simulation behavior (time steps, model files, noise, etc.)
-   **Reward Configuration** affects learning signals (reward weights, calculation methods, etc.)
-   **Training Configuration** controls algorithm behavior (network structure, learning rate, batch size, etc.)

## Multi-Framework Support

MotrixLab's layered design naturally supports multiple RL frameworks:

-   **Simulation Backends**: MotrixSim (CPU)
-   **Training Frameworks**:
    -   **SKRL**: Supports JAX and PyTorch backends with GPU acceleration
    -   **RSLRL**: Supports PyTorch backend with GPU acceleration
-   **Framework Selection**: Use `--rllib` parameter to choose between `skrl` (default) and `rslrl`

## Design Advantages

This architecture design brings the following core advantages:

1. **Module Decoupling**: Environment development and training logic are completely separated
2. **Flexible Configuration**: Supports hierarchical configuration and runtime parameter override
3. **Strong Extensibility**: Easily add new components through the registry system
4. **Multi-Backend Compatibility**: Same environment can use different simulation and training backends
5. **Experiment-Friendly**: Configurations can be saved and compared, ensuring experimental reproducibility

Through this framework design, MotrixLab provides a clear, flexible, and easy-to-use development platform for robot reinforcement learning.
