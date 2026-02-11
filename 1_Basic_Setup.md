# 基础环境搭建指南 (Basic Setup)

本指南介绍如何在 Windows 上配置基础的 Linux 开发环境，包含避免 C 盘爆满的技巧。

## 1. WSL2 环境准备与离线安装

### 1.1 准备工作目录
首先，在非系统盘（如 E 盘）创建一个专门存放 WSL 数据的目录，防止后续系统占用过多 C 盘空间：
```powershell
mkdir E:\WSL
```

### 1.2 下载与提取系统镜像
使用 PowerShell 下载 Ubuntu 22.04 的官方发行包：
```powershell
Invoke-WebRequest -Uri https://aka.ms/wslubuntu2204 -OutFile E:\WSL\ubuntu2204.zip
```
**提取 `install.tar.gz` (关键步骤)**:
- 解压下载的 `ubuntu2204.zip`。
- 在内部找到 `Ubuntu_2204.x.x.x_x64.appx`，将其后缀手动改为 `.zip` 并再次解压。
- 在第二次解压的目录里，你就能找到最终需要的 `install.tar.gz` 根文件系统。

### 1.3 导入并运行 WSL
使用 `wsl --import` 命令将系统安装到你指定的非系统盘路径：
```powershell
# 导入系统：wsl --import <自定义名称> <安装位置> <镜像文件路径>
wsl --import Ubuntu-22.04 E:\WSL\Ubuntu2204_Data E:\WSL\path\to\install.tar.gz

# 启动并进入系统
wsl -d Ubuntu-22.04
```

## 2. 用户权限配置 (安全最佳实践)

默认导入的系统是以 `root` 身份登录的。为了开发安全，我们需要创建一个标准用户并赋予 `sudo` 权限：

```bash
# 添加新用户并按提示设置密码
adduser cvuser

# 将用户添加到 sudo 组，使其拥有管理员权限
usermod -aG sudo cvuser

# 退出并设置默认登录用户 (在 Windows PowerShell 执行以下命令)
ubuntu2204 config --default-user cvuser
```

## 3. 基础开发工具安装

进入 WSL2 后，安装 C++ 编译环境和 OpenCV 基础库：

### 3.1 编译器与构建工具
安装 `build-essential`（包含 g++, make）和 `cmake`：
```bash
sudo apt update && sudo apt install build-essential cmake -y
```

### 3.2 安装 OpenCV
通过 Ubuntu 官方仓库安装 OpenCV 开发库，这会自动配置好相关的依赖项：
```bash
sudo apt install libopencv-dev -y
```
