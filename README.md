# ⚡ CV-CUDA & TensorRT: Full-Pipeline Acceleration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Technology: CV-CUDA](https://img.shields.io/badge/Tech-CV--CUDA-green)](https://github.com/CVCUDA/CV-CUDA)
[![Technology: TensorRT](https://img.shields.io/badge/Tech-TensorRT-blue)](https://developer.nvidia.com/tensorrt)

[**English**](./README.md) | [**中文**](./README.md)

## 🚀 项目简介 / Introduction

在高性能 AI 推理场景中，**预处理（Preprocessing）** 往往是制约系统吞吐量的核心瓶颈。即便 TensorRT 推理只需 5ms，但 CPU 上的图像解码、缩放、重排（HWC->NCHW）可能需要 30ms。

本项目通过实测对比，展示了如何利用 NVIDIA **CV-CUDA** 的融合算子（Fused Operators）完全消除 CPU 瓶颈，实现预处理与推理的纯 GPU 流水线衔接。

In high-performance AI inference, **preprocessing** is often the bottleneck. This project demonstrates how to eliminate CPU overhead by using NVIDIA **CV-CUDA** fused operators, creating a pure GPU pipeline from raw image to TensorRT inference.

## 🛠️ 核心加速技术 / Key Features

- **Standard OpenCV vs. CV-CUDA Accelerated**: 深度对比标准 OpenCV 处理流程与 CV-CUDA 硬件加速版本。
- **Fused Operators (融合算子)**: 使用 `ResizeCropConvertReformat` 将裁切、归一化、排列重组合并为单次 Kernel 调用。
- **D2D Gather (显存内收集)**: 利用 `cudaMemcpy2DAsync` 实现显存内 ROI 快速并行提取，规避 PCIe 带宽瓶颈。
- **TensorRT 10.10 Integration**: 衔接最新的 TensorRT 10.10 `enqueueV3` 接口，展示端到端的推理优化。

## 📊 性能对标 / Benchmark

**测试环境：** 4480x4480 大图 -> 224x224 切片 x 400 张 (Batch Size = 25)

| 方案 / Method | 预处理技术 / Technology | 耗时 / Latency | 吞吐量提升 / Speedup |
| :--- | :--- | :--- | :--- |
| **Method A** | Standard OpenCV (SIMD Optimized) | ~32.3 ms | Baseline |
| **Method B** | **CV-CUDA Accelerated (Fused Batch)** | **~7.5 ms** | **⚡ 4.3x Faster** |

> *注：耗时包含数据从 CPU 上传到 GPU 的全过程。随着切片数量增加，CV-CUDA 的优势将进一步扩大。*

## 📂 项目结构 / Structure

- `trt_preprocessing_benchmark.cpp`: **[核心]** 预处理对比与端到端推理测试。
- `cvcuda_fair_benchmark.cpp`: 早期公平性测试对比代码。
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
