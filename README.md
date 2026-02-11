# ⚡ CV-CUDA & TensorRT: Full-Pipeline Acceleration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Technology: CV-CUDA](https://img.shields.io/badge/Tech-CV--CUDA-green)](https://github.com/CVCUDA/CV-CUDA)
[![Technology: TensorRT](https://img.shields.io/badge/Tech-TensorRT-blue)](https://developer.nvidia.com/tensorrt)

[**English**](./README.md) | [**中文**](./README.md)

## 🚀 项目简介 / Introduction

在高性能 AI 推理场景中，**预处理（Preprocessing）** 往往是制约系统吞吐量的核心瓶颈。本项目通过实测对比，展示了如何利用 NVIDIA **CV-CUDA** 的融合算子（Fused Operators）极大程度消除 CPU 瓶颈，实现从原始图像到推理结果的纯 GPU 高速流水线。

In high-performance AI inference, **preprocessing** is often the bottleneck. This project demonstrates how to effectively bypass CPU overhead by using NVIDIA **CV-CUDA** fused operators, creating a high-throughput GPU-centric pipeline.

## 💻 测试平台 / Test Platform

- **CPU**: Intel Core i7-12700F
- **GPU**: NVIDIA GeForce RTX 4070 (12GB)
- **Software**: CUDA 12.x, TensorRT 10.10, CV-CUDA 0.x

## 🛠️ 核心加速技术 / Key Features

- **Standard OpenCV vs. CV-CUDA Accelerated**: 深度对比标准 OpenCV 处理流程与 CV-CUDA 硬件加速版本。
- **Fused Operators (融合算子)**: 使用 `ResizeCropConvertReformat` 将裁切、归一化、排列重组合并为单次 Kernel 调用。
- **D2D Gather (显存内收集)**: 利用 `cudaMemcpy2DAsync` 实现显存内 ROI 快速并行提取，规避 PCIe 带宽瓶颈。
- **TensorRT 10.10 Integration**: 衔接最新的 TensorRT 10.10 `enqueueV3` 接口。

## 📊 性能对标 / Benchmark

**测试环境：** 4480x4480 大图 -> 224x224 切片 x 400 张 (Batch Size = 25)

| 方案 / Method | 预处理技术 / Technology | 耗时 / Latency | 吞吐量提升 / Speedup |
| :--- | :--- | :--- | :--- |
| **Method A** | Standard OpenCV (SIMD Optimized) | ~32.3 ms | Baseline |
| **Method B** | **CV-CUDA Accelerated (Fused Batch)** | **~7.5 ms** | **⚡ 4.3x Faster** |

## 📂 项目结构 / Structure

- `trt_preprocessing_benchmark.cpp`: **[核心]** 预处理对比与端到端推理测试。
- `hello_world.cpp`: CV-CUDA 入门示例。
- `examples/`:
  - `opencv_cvcuda_comparison.cpp`: 基础算子（Crop/Resize等）的性能对比示例。
- `1_Basic_Setup.md`: 环境搭建指南。
- `2_CUDA_And_CV-CUDA_Setup.md`: 深度优化配置参考。

## 🛠️ 编译运行 / Build & Run

```bash
mkdir build && cd build
cmake ..
make
./my_app
```

---
*If you find this project helpful, please give it a ⭐! 如果这个项目对你有帮助，请点个 Star！*
