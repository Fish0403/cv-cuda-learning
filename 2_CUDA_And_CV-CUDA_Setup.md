# CUDA, CV-CUDA & TensorRT 安装指南

本指南详细说明如何配置 NVIDIA 相关的深度加速环境，并包含关键的“避坑指南”。

## 🚨 核心警告：不要在 WSL2 中安装 NVIDIA 驱动
WSL2 特有的 GPU 穿透技术直接复用 Windows 主机上的驱动。**切勿**在 WSL2 内部安装任何 `.run` 或 `.deb` 格式的 NVIDIA 驱动，否则会破坏系统内核链接。

## 1. CUDA Toolkit 安装

### 1.1 检查驱动版本选择 Toolkit
在终端运行 `nvidia-smi`，根据右上角的 `CUDA Version` 限制选择对应的 Toolkit 版本。建议版本匹配以获得最佳稳定性。

### 1.2 安装步骤 (以 12.8 为例)
通过 NVIDIA 官方仓库安装，这样后续可以通过 `apt upgrade` 方便地更新：
```bash
# 下载并安装 NVIDIA 仓库密钥环
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 更新索引并安装工具包主体
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### 1.3 环境变量配置 (必须)
为了让编译器和系统能找到 CUDA 的二进制文件和动态链接库，必须配置环境变量。编辑 `~/.bashrc`：
```bash
# 在文件末尾添加以下路径
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LIBRARY_PATH=$CUDA_PATH/lib64/stubs:$CUDA_PATH/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```
添加完成后，运行 `source ~/.bashrc` 使其立即生效。

## 2. CV-CUDA 安装

### 2.1 下载 Debian 包
CV-CUDA 提供了预编译的 `.deb` 包。请从 [CV-CUDA Release](https://github.com/CVCUDA/CV-CUDA/releases) 页面下载 runtime 库和 dev 开发头文件：
```bash
# 下载示例 (版本号请根据实际情况调整)
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.16.0/cvcuda-lib-0.16.0-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.16.0/cvcuda-dev-0.16.0-cuda12-x86_64-linux.deb
```

### 2.2 执行安装与验证
使用 `apt install` 安装本地包，它会自动处理可能缺失的依赖：
```bash
sudo apt update && sudo apt install -y ./cvcuda-lib-*.deb ./cvcuda-dev-*.deb

# 验证安装：查看头文件是否已正确放置在系统目录
ls /usr/include/cvcuda/
```

## 3. TensorRT 10.10 安装

### 3.1 导入密钥与更新 (最关键)
TensorRT 的本地仓库包安装后，必须手动将 GPG 密钥拷贝到系统受信列表，否则 `apt update` 会因为无法验证签名而报错：
```bash
# 安装本地仓库定义文件
sudo dpkg -i tensorrt_10.10.deb

# 拷贝 GPG 密钥（注意文件夹名需根据实际版本补全）
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.10.0-cuda-12.9/*-keyring.gpg /usr/share/keyrings/

# 更新索引并正式安装 TensorRT 主程序
sudo apt-get update
sudo apt-get install tensorrt
```

### 3.2 trtexec 工具配置
`trtexec` 是 TensorRT 最常用的性能测试和模型转换工具，默认不在系统路径中。我们可以将其加入 PATH：
```bash
# 将二进制路径永久加入环境变量
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc
```

## 4. 环境最终验证
完成所有步骤后，运行以下命令确保环境通畅：
- **GPU 穿透**: `nvidia-smi`（应显示显卡信息）
- **CUDA 编译器**: `nvcc --version`（应显示版本号）
- **TensorRT**: `python3 -c "import tensorrt; print(tensorrt.__version__)"`
