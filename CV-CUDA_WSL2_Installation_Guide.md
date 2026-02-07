# CV-CUDA on WSL2 安装指南

### 避坑指南：WSL2 中的 NVIDIA 驱动

本指南将引导您如何搭建用于 CV-CUDA 开发的 WSL2 环境。切勿在 WSL2 中安装 Linux 版 NVIDIA 驱动，请务必使用不含驱动的 NVIDIA WSL-Ubuntu 专用 CUDA 工具包，以避免冲突。

## 第一部分：WSL2 CUDA 环境准备与配置

### WSL2 环境准备与配置

1.  **创建 WSL 目录：** 在您的目标磁盘（例如 E:）上，创建一个目录用于存放 WSL 相关数据。
    ```powershell
    mkdir E:\WSL
    ```

2.  **下载 Ubuntu 镜像：** 以管理员身份打开 PowerShell，下载 Ubuntu 22.04 的离线安装包。
    ```powershell
    Invoke-WebRequest -Uri https://aka.ms/wslubuntu2204 -OutFile E:\WSL\ubuntu2204.zip
    ```

3.  **提取根文件系统：** `install.tar.gz` 文件被封装在多层压缩包内，请按照以下步骤提取：

    -   **第一步：** 解压 `ubuntu2204.zip`。

    -   **第二步：** 在解压后的文件夹中，找到名为 `Ubuntu_2204.x.x.x_<架构>.appx` 的文件。

        **重要提示：** 根据您的 CPU 架构（x64 或 arm64）选择正确的文件，并将其后缀名改为 `.zip`，然后再次解压。

        **错误选择架构会导致 `Exec format error` 报错。**

    -   **第三步：** 在第二次解压后的文件夹中，您会找到 `install.tar.gz` 文件。请记下它的完整路径：`E:\WSL\ubuntu2204\install.tar.gz`。

### 在非系统盘上安装 WSL2

1.  **创建目标目录：** 在 PowerShell 中，创建最终存放 Ubuntu 系统的目录。
    ```powershell
    mkdir E:\WSL\Ubuntu2204_Data
    ```

2.  **导入发行版：** 使用 `wsl --import` 命令将 Ubuntu 安装到您指定的路径。
    ```powershell
    # wsl --import <自定义名称> <安装位置> <tar.gz文件路径>
    wsl --import Ubuntu-22.04 E:\WSL\Ubuntu2204_Data E:\WSL\ubuntu2204\install.tar.gz
    ```

### 用户配置

默认情况下，导入的系统以 `root` 用户登录。为安全起见并遵循开发最佳实践，我们来创建一个标准用户。

1.  **进入 WSL 实例：**
    ```powershell
    wsl -d Ubuntu-22.04
    ```

2.  **创建新用户：** 在 Ubuntu 环境中，运行以下命令（请将 `cvuser` 替换为您想要的用户名）。
    ```bash
    # 创建用户
    adduser cvuser

    # 将用户添加到 sudo 组以获取管理员权限
    usermod -aG sudo cvuser
    ```

3.  **设置默认用户：** 输入 `exit` 退出 Ubuntu Shell 返回到 PowerShell。然后，设置您的新用户为此 WSL 发行版的默认用户。
    ```powershell
    ubuntu2204 config --default-user cvuser
    ```

### CUDA Toolkit 安装 (WSL2 环境内)

WSL2 使用安装在 Windows 主机上的 NVIDIA 驱动程序。因此，请确保不要在 WSL2 发行版中安装 NVIDIA 驱动程序。有关 WSL2 和使用 CUDA 的更多信息，请访问[CUDA 工具包 - WSL2页面](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

在安装 CUDA Toolkit 之前，请通过 `nvidia-smi` 命令查看您的 NVIDIA 驱动版本。
```bash
nvidia-smi
```
根据驱动版本选择对应的 CUDA Toolkit：
-   如果您使用的是 r580 (即 580.xx) 或更高版本的驱动程序，则需要安装 CUDA 工具包 13.0。
-   如果您的驱动程序版本为 r525 (即 525.xx) 或更高版本，则需要安装 CUDA 工具包 12.8。

1.  **适用于 Linux WSL-Ubuntu 2.0 x86_64 的安装说明**
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    ```

2.  **设置环境变量：**
    安装好 CUDA Toolkit 后，需要在 WSL2 发行版中设置以下环境变量。您可以使用这些命令自动更新 `~/.bashrc` 文件中的环境变量。
    ```bash
    echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc
    echo 'export LIBRARY_PATH=$CUDA_PATH/lib64/stubs:$CUDA_PATH/lib64:$LIBRARY_PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    ```
    然后执行 `source ~/.bashrc` 使其生效。

4.  **验证 CUDA 安装：**
    ```bash
    nvcc --version
    ```
    应该显示 CUDA Compiler Driver release 12.8, V12.8.xx.

### 环境验证

重新进入您的 WSL 实例，此时应该会以您的新用户身份登录。

```powershell
wsl -d Ubuntu-22.04
```

1.  **验证 GPU 穿透：** 这是最关键的一步。
    ```bash
    nvidia-smi
    ```
    如果此命令成功显示您的 NVIDIA GPU 信息，则表示您的 GPU 已正确穿透到 WSL2 环境。

2.  **验证 CUDA Toolkit 安装：**
    ```bash
    nvcc --version
    ```
    应该显示 CUDA Compiler Driver release 12.8, V12.8.xx。

3.  **验证内核版本：**
    ```bash
    uname -r
    ```
    请确保内核版本符合 NVIDIA 驱动程序的要求。

## 第二部分：安装 CV-CUDA

完成上述步骤后，即可在 WSL2 发行版中安装 CV-CUDA。CV-CUDA 提供多种安装方式，主要包括：

*   **从 PyPI 安装:** 推荐用于 Python 用户。详情请参考 [PyPI 上的 Python Wheels](https://pypi.org/project/nvidia-cvcuda/)。
*   **从预编译软件包安装:** 包括 Debian 或 Tar 归档文件。详情请参阅 [Debian 软件包和 Tar 归档文件](https://docs.nvidia.com/cvcuda/install.html#installing-from-deb-packages-and-tar-archives)。
*   **从源代码构建:** 适用于需要自定义编译的用户。请查阅官方 [从源代码构建](https://docs.nvidia.com/cvcuda/install.html#installing-from-source) 说明。

本指南接下来将详细介绍**从预编译 Debian 软件包安装**的方法。

#### 🛠️ 安装步骤

为了方便管理和安装，建议在 WSL2 的 Linux 环境中执行以下操作：

##### 1. 准备工作目录

首先，回到你的 Linux 家目录，并创建一个专门存放 CV-CUDA 项目的文件夹。

```bash
cd ~
mkdir -p projects/my_cvcuda_project
cd projects/my_cvcuda_project
# 此时，你的当前路径应该类似：/home/你的用户名/projects/my_cvcuda_project
pwd
```

##### 2. 下载 Debian 包

从 [CV-CUDA v0.16.0 Release](https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.16.0) 下载以下包。请根据您的 CUDA 版本和系统架构选择正确的文件。

如果您需要其他版本，可以访问 [CV-CUDA GitHub 发布页面](https://github.com/NVIDIA/CV-CUDA/releases) 进行下载。

```bash
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.16.0/cvcuda-lib-0.16.0-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.16.0/cvcuda-dev-0.16.0-cuda12-x86_64-linux.deb
```

##### 3. 执行安装

下载完成后，在当前目录下执行安装：

```bash
sudo apt update && sudo apt install -y ./cvcuda-lib-*.deb ./cvcuda-dev-*.deb
```

##### 4. 验证 CV-CUDA 安装

安装完成后，您可以通过检查 CV-CUDA 的头文件目录来验证其是否成功安装。

```bash
ls /usr/include/cvcuda/
```

如果此命令列出了文件和目录（例如 `cvcuda` 文件夹内应包含头文件等），则表明 CV-CUDA 已成功安装。
