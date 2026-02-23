# ⚡ CV-CUDA & TensorRT: Full-Pipeline Acceleration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Technology: CV-CUDA](https://img.shields.io/badge/Tech-CV--CUDA-green)](https://github.com/CVCUDA/CV-CUDA)
[![Technology: TensorRT](https://img.shields.io/badge/Tech-TensorRT-blue)](https://developer.nvidia.com/tensorrt)

[**English**](./README.md) | [**中文**](./README.md)

## 🚀 项目简介 / Introduction

在高性能 AI 推理场景中，**预处理（Preprocessing）** 往往是制约系统吞吐量的核心瓶颈。本项目聚焦 OpenCV CPU、OpenCV CUDA、CV-CUDA 三种实现的实测对比，覆盖算子级和流程级性能，并明确区分 **Kernel 时间** 与 **端到端时间（含 H2D/D2H）**。

In high-performance AI inference, **preprocessing** is often the bottleneck. This project benchmarks OpenCV CPU, OpenCV CUDA, and CV-CUDA at both operator and pipeline levels, and separates **kernel time** from **end-to-end latency (including H2D/D2H transfers)**.

## 💻 测试平台 / Test Platform

- **CPU**: Intel Core i7-12700F
- **GPU**: NVIDIA GeForce RTX 4070 (12GB)
- **Software**: Ubuntu 22.04, CUDA 12.x, OpenCV 4.11, TensorRT 10.10, CV-CUDA 0.x

## 🛠️ 核心加速技术 / Key Features

- **Standard OpenCV vs. CV-CUDA Accelerated**: 深度对比标准 OpenCV 处理流程与 CV-CUDA 硬件加速版本。
- **Fused Operators (融合算子)**: 使用 `ResizeCropConvertReformat` 将裁切、归一化、排列重组合并为单次 Kernel 调用。
- **D2D Gather (显存内收集)**: 利用 `cudaMemcpy2DAsync` 实现显存内 ROI 快速并行提取，规避 PCIe 带宽瓶颈。
- **TensorRT 10.10 Integration**: 衔接最新的 TensorRT 10.10 `enqueueV3` 接口。

## 📊 性能对标 / Benchmark

### 1) 典型推理流程（Preprocess + Inference）三者对比

**测试条件（单次实测）：**
- 输入图：`4480x4480`（`224x224` 网格切片，共 `400` patches）
- 预处理批次：`batch_size=25`（共 `16` 个 batch）
- 模型：`model.onnx -> model.engine`（动态输入，`min=1x3x224x224`, `opt=25x3x224x224`, `max=96x3x224x224`）
- 统计口径：下表时间均为**处理完 400 张 patch 的总预处理时间**（非单 batch 时间）

| 方案 / Method | 预处理技术 / Technology | 耗时 / Latency | 吞吐量提升 / Speedup |
| :--- | :--- | :--- | :--- |
| **Method A** | Standard OpenCV (SIMD Optimized) | 40.0936 ms | Baseline |
| **Method B** | OpenCV CUDA Pipeline (Non-Fused) | 27.3381 ms | 1.47x |
| **Method C** | **CV-CUDA Accelerated (Fused Batch)** | **7.6871 ms** | **5.22x** |

### 2) 算子级对比（examples）

#### `op_average_blur` 三者时间对比（单次实测）

**配置：** `Image=5120x5120x1`, `Kernel=7x7`, `warmup=3`, `iters=10`

| 方法 | H2D (ms) | Kernel Benchmark (ms) | D2H (ms) | Total (ms) |
| :--- | :---: | :---: | :---: | :---: |
| OpenCV CPU | N/A | 130.327 | N/A | 130.327 |
| OpenCV CUDA | 5.9028 | 24.8877 | 14.7654 | 45.5559 |
| CV-CUDA | 5.3105 | 22.7692 | 15.1054 | 43.1851 |

#### `op_resize` 三者时间对比

**配置：** `Batch=1`, `5120x5120 -> 4480x4480`, `warmup=3`

| 方法 | H2D (ms) | Kernel Benchmark (ms) | D2H (ms) | Total (ms) |
| :--- | :---: | :---: | :---: | :---: |
| OpenCV CPU | N/A | 8.5855 | N/A | 8.5855 |
| OpenCV CUDA | 14.3371 | 0.3861 | 25.5832 | 40.3064 |
| CV-CUDA | 15.9585 | 0.3594 | 33.8105 | 50.1284 |

#### `op_warp_affine` 三者时间对比

**配置：** `Image=8200x6000(gray)`, `angle=5 deg`, `warmup=3`

| 方法 | H2D (ms) | Kernel Benchmark (ms) | D2H (ms) | Total (ms) |
| :--- | :---: | :---: | :---: | :---: |
| OpenCV CPU | N/A | 14.3759 | N/A | 14.3759 |
| OpenCV CUDA | 10.4717 | 0.6265 | 28.0645 | 39.1627 |
| CV-CUDA | 11.5325 | 3.1897 | 27.6091 | 42.3313 |

## 📂 项目结构 / Structure

- `trt_preprocessing_benchmark.cpp`: **[核心]** 预处理对比与端到端推理测试。
- `hello_world.cpp`: CV-CUDA 入门示例。
- `examples/`:
  - `op_resize.cpp`: OpenCV CPU / OpenCV CUDA / CV-CUDA 的 Resize 对比。
  - `op_average_blur.cpp`: OpenCV CPU / OpenCV CUDA / CV-CUDA 的均值模糊对比。
  - `op_warp_affine.cpp`: OpenCV CPU / OpenCV CUDA / CV-CUDA 的仿射变换对比。
- `1_Basic_Setup.md`: 环境搭建指南。
- `2_CUDA_And_CV-CUDA_Setup.md`: 深度优化配置参考。

## 🛠️ 编译运行 / Build & Run

```bash
mkdir build && cd build
cmake ..
make
./op_resize
./op_average_blur
./op_warp_affine
```

---
*If you find this project helpful, please give it a ⭐! 如果这个项目对你有帮助，请点个 Star！*
