# 安装环境

本文档将引导您完成 MotrixLab 的安装与配置。请仔细阅读系统要求，并根据您的使用场景选择合适的安装方式。

## 系统要求

-   **Python 版本**：{bdg-danger-line}`3.10.*`

    本项目依赖特定 Python 版本，其他版本暂不受支持：

    | Python 版本 | 支持状态 |
    | :---------: | :------: |
    |    ≤ 3.9    |    ❌    |
    |    3.10     |    ✅    |
    |   ≥ 3.11    |    ❌    |

-   **包管理器**：{bdg-danger-line}`UV`

    本项目采用 UV 作为唯一的包管理工具，以提供快速、可复现的依赖管理环境。UV 的安装方法请参考[官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

-   **系统及架构**：

    -   {bdg-danger-line}`Windows(x86_64)`
    -   {bdg-danger-line}`Linux(x86_64)`

    ```{note}
    不同操作系统对 MotrixLab 各功能模块的支持情况如下：

    | 操作系统 | CPU 仿真 | 交互式查看器 | GPU 仿真 |
    | :------: | :------: | :----------: | :------: |
    |  Linux   |    ✅    |      ✅      |    🛠️ 开发中    |
    | Windows  |    ✅    |      ✅      |    🛠️ 开发中    |
    ```

## 安装步骤

### 克隆项目仓库

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

### 配置依赖环境

:::{dropdown} 配置国内镜像源（可选）
:animate: fade-in
:color: warning
:icon: desktop-download
如果您身处中国大陆，建议配置国内镜像源以加速依赖下载：

1. 修改项目根目录的 `uv.toml` 文件

    ```toml
    [[index]]
    name = "mirror"
    # 请填写您选择的国内镜像源，例如：
    # 清华源: "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    url = ""


    [[index]]
    name = "pytorch"
    url = "https://download.pytorch.org/whl/cu128"
    default = true
    ```

2. 在执行 `uv sync` 命令时添加 `--index-strategy unsafe-best-match` 参数：

    ```
    uv sync --all-packages --all-extras --index-strategy unsafe-best-match
    ```

:::

执行以下命令安装完整依赖：

```bash
# 安装所有依赖
uv sync --all-packages --all-extras
```

如果仅需特定训练框架，可选择性安装以减少依赖体积：

```bash

# 安装 SKRL JAX （仅支持 Linux 平台）
uv sync --all-packages --extra skrl-jax

# 安装 SKRL PyTorch
uv sync --all-packages --extra skrl-torch

# 安装 RSLRL（仅支持 PyTorch）
uv sync --all-packages --extra rslrl
```
