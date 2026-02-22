# OpenCV CUDA 安装与编译指南

开始前需完成 `2_CUDA_And_CV-CUDA_Setup.md` 中的 `1. CUDA Toolkit 安装`，并确保 `nvidia-smi`、`nvcc --version` 均可正常输出。

## 1. cuDNN（CUDA 12.8 示例：cuDNN 9.19）

- 只做 OpenCV CUDA 基础算子对比可不装 cuDNN。
- 使用 OpenCV DNN CUDA（`cv::dnn` CUDA 后端）时，需要安装 cuDNN。
- 本节以 CUDA 12.8 + cuDNN 9.19（Ubuntu 22.04 x86_64）为例。

先安装 zlib：

```bash
sudo apt-get install zlib1g
```

再安装 cuDNN（Local Installer 配置 + CUDA 12 包）：

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.19.0/local_installers/cudnn-local-repo-ubuntu2204-9.19.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.19.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.19.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12
```

说明：
- 升级到 CUDA 13 后，对应安装 `cudnn9-cuda-13`。

验证：

```bash
sudo ldconfig
ldconfig -p | grep -i cudnn
dpkg -l | grep -i cudnn
```
## 2. 安装编译依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config \
  libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
  libtbb-dev libjpeg-dev libpng-dev libtiff-dev libopenexr-dev \
  libxvidcore-dev libx264-dev libv4l-dev v4l-utils \
  libdc1394-dev libopenblas-dev liblapack-dev gfortran \
  python3-dev python3-numpy
```

## 3. 拉取 OpenCV 源码（含 contrib）

```bash
mkdir -p ~/third_party/opencv-4.11
cd ~/third_party/opencv-4.11
git clone --branch 4.11.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.11.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

说明：`opencv` 与 `opencv_contrib` 版本号需保持一致。

## 4. 配置 CMake（开启 CUDA）

```bash
cd ~/third_party/opencv-4.11/opencv
mkdir -p build && cd build

CLEAN_PATH=$(printf "%s" "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/msys64' | paste -sd:)
env PATH="$CLEAN_PATH" cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=~/third_party/opencv-4.11/opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=8.9 \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_OPENEXR=OFF \
  -D WITH_OPENJPEG=OFF \
  -D CMAKE_IGNORE_PREFIX_PATH=/mnt/c/msys64/mingw64 \
  -D CMAKE_IGNORE_PATH=/mnt/c/msys64/mingw64 \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D WITH_CUBLAS=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF
```

说明：
- `CUDA_ARCH_BIN=8.9` 适配 RTX 4070（Ada）。
- 其他显卡需改成对应算力（如 8.6、8.0、7.5 等）。
- `CLEAN_PATH` 用于排除 `/mnt/c/msys64`，避免混入 Windows 头文件与库。
- `WITH_OPENEXR=OFF`、`WITH_OPENJPEG=OFF` 用于规避当前环境中的跨平台依赖冲突。
- 不需要 OpenCV DNN CUDA 时，可改为 `-D WITH_CUDNN=OFF -D OPENCV_DNN_CUDA=OFF`，并跳过第 1 节 cuDNN 安装。

## 5. 编译与安装

```bash
nproc
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 6. 验证是否真的是 CUDA 版 OpenCV

```bash
opencv_version
opencv_version --verbose | grep -i cuda
```

应看到 `CUDA: YES` 相关信息。
