# CV-CUDA C++ 编译与运行指南

本文档主要介绍如何在 WSL 环境下配置 C++ 编译环境，链接 OpenCV 库并运行代码。

> **前提**：假设你已经完成了 NVIDIA 驱动和 CUDA Toolkit 的安装。

## 1. 安装编译工具

编译 C++ 代码需要 `g++` 编译器：

```bash
sudo apt update
sudo apt install build-essential
```

## 2. 安装 OpenCV （可选）
在 Ubuntu 22.04 上，最简单且推荐的方法是通过官方仓库安装。

```bash
sudo apt update
sudo apt install libopencv-dev
```

## 3. 编译 (使用 CMake)

推荐使用 CMake 进行管理，比手动输入 `g++` 命令更规范。

### 3.1 准备
确保目录下已有 `CMakeLists.txt` 文件。

### 3.2 编译

```bash
# 创建构建目录并编译
mkdir -p build && cd build
cmake ..
make
```

## 4. 运行

直接运行即可 (CMake 已自动配置好路径)：

```bash
./hello_world
```