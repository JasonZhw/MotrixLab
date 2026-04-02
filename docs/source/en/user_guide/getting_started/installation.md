# Installation Environment

This document will guide you through the installation and configuration of MotrixLab. Please read the system requirements carefully and choose the appropriate installation method based on your use case.

## System Requirements

-   **Python Version**: {bdg-danger-line}`3.10.*`

    This project requires a specific Python version, other versions are not currently supported:

    | Python Version | Support Status |
    | :------------: | :------------: |
    |     ≤ 3.9      |       ❌       |
    |      3.10      |       ✅       |
    |     ≥ 3.11     |       ❌       |

-   **Package Manager**: {bdg-danger-line}`UV`

    This project uses UV as the exclusive package management tool to provide fast, reproducible dependency management environment. For UV installation, please refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

-   **System and Architecture**:

    -   {bdg-danger-line}`Windows(x86_64)`
    -   {bdg-danger-line}`Linux(x86_64)`

    ```{note}
    Features supported on each platform:

    | Operating System | CPU Simulation | Interactive Viewer | GPU Simulation |
    | :--------------: | :------------: | :----------------: | :------------: |
    |      Linux       |       ✅       |         ✅          |    🛠️ In Development    |
    |     Windows      |       ✅       |         ✅          |    🛠️ In Development    |
    ```

## Installation Steps

### Clone Project Repository

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

### Configure Dependencies

Execute the following command to install complete dependencies:

```bash
# Install all dependencies
uv sync --all-packages --all-extras
```

If you only need specific training frameworks, you can selectively install to reduce dependency size:

```bash

# Install SKRL JAX (Linux only)
uv sync --all-packages --extra skrl-jax

# Install SKRL PyTorch
uv sync --all-packages --extra skrl-torch

# Install RSLRL (PyTorch only)
uv sync --all-packages --extra rslrl
```
