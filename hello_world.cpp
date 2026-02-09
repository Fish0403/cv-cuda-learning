/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <iostream>
 #include <vector>
 #include <chrono>
 #include <filesystem>
 
 // --- 严格按照你的头文件路径 ---
 #include <cvcuda/OpGaussian.hpp>
 #include <cvcuda/OpResize.hpp>
 #include <cvcuda/OpStack.hpp>
 #include <nvcv/Image.hpp>
 #include <nvcv/Tensor.hpp>
 #include <cvcuda/cuda_tools/TypeTraits.hpp>
 #include <nvcv/TensorBatch.hpp>
 // CUDA 运行时
 #include <cuda_runtime.h>
 
 namespace fs = std::filesystem;
 
 // 计时器类 (模仿 Python timer)
 class Timer {
    public:
        Timer(const std::string& tag) : m_tag(tag), m_start(std::chrono::high_resolution_clock::now()) {}
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            // 修正点：使用 std::milli 而不是 std::m_unit::milli
            std::chrono::duration<double, std::milli> diff = end - m_start;
            std::cout << m_tag << ": \n  Time: " << diff.count() << " ms\n--------------------------------" << std::endl;
        }
    private:
        std::string m_tag;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    };
 
 int main(int argc, char* argv[]) {
     // 基础配置
     int batch_size = 2; // 模仿 Python 的多图处理
     int target_w = 224, target_h = 224;
     int kernel_size = 5;
     float sigma = 1.0f;
 
     cudaStream_t stream;
     cudaStreamCreate(&stream);
 
     try {
         // 1. 模拟加载图像 (Load Images)
         // 对应 Python: tensors = [cvcuda.as_tensor(image, "HWC") for image in images]
         std::vector<nvcv::Tensor> input_tensors;
         {
             Timer t("Step 1: Create Input Tensors");
             for (int i = 0; i < batch_size; ++i) {
                 // 创建 640x480 的原始张量
                 input_tensors.emplace_back(nvcv::Tensor({{1, 480, 640, 3}, "NHWC"}, nvcv::TYPE_U8));
             }
         }
 
         // 2. 缩放图像 (Resize)
         // 对应 Python: resized_tensors = [cvcuda.resize(...) for tensor in tensors]
         std::vector<nvcv::Tensor> resized_tensors;
         {
             Timer t("Step 2: Resize Images");
             cvcuda::Resize resizeOp; 
             for (int i = 0; i < batch_size; ++i) {
                 resized_tensors.emplace_back(nvcv::Tensor({{1, target_h, target_w, 3}, "NHWC"}, nvcv::TYPE_U8));
                 resizeOp(stream, input_tensors[i], resized_tensors[i], NVCV_INTERP_LINEAR);
             }
         }
 
        // 3. 堆叠成批 (Batching via Stack)
        nvcv::Tensor batch_tensor({{batch_size, target_h, target_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        {
            Timer t("Step 3: Stack into Batch");

            // 使用 nvcv::TensorBatch 代替之前的 VarShape 类
            // 通常构造函数接受最大张量数量
            nvcv::TensorBatch batchContainer(batch_size);

            // 将 std::vector 中的 tensor 逐个推入
            for (auto &t : resized_tensors) {
                batchContainer.pushBack(t);
            }

            // 调用 stackOp
            cvcuda::Stack stackOp;
            stackOp(stream, batchContainer, batch_tensor);
        }
         // 4. 高斯模糊 (Apply Gaussian blur)
         // 对应 Python: blurred_tensor_batch = cvcuda.gaussian(...)
         nvcv::Tensor blurred_batch({{batch_size, target_h, target_w, 3}, "NHWC"}, nvcv::TYPE_U8);
         {
             Timer t("Step 4: Gaussian Blur on Batch");
             // 注意：按照你之前的报错，这里需要传入最大内核大小和批次大小
             cvcuda::Gaussian gaussianOp({kernel_size, kernel_size}, batch_size);
             gaussianOp(stream, batch_tensor, blurred_batch, 
                        {kernel_size, kernel_size}, {sigma, sigma}, NVCV_BORDER_CONSTANT);
         }
 
         // 5. 同步与结束
         cudaStreamSynchronize(stream);
         std::cout << "✅ 模仿 Python 流程的 C++ Batch 示例执行完毕！" << std::endl;
 
     } catch (const std::exception& e) {
         std::cerr << "❌ 运行时错误: " << e.what() << std::endl;
         return -1;
     }
 
     cudaStreamDestroy(stream);
     return 0;
 }