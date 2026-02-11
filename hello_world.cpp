/*
 * Simple application showing how to build a pipeline with CV-CUDA in C++.
 * 
 * This application demonstrates:
 * 1. Loading images into GPU memory.
 * 2. Resizing images using CV-CUDA.
 * 3. Batching multiple images into a single Tensor.
 * 4. Applying Gaussian Blur on the batched Tensor.
 * 5. Saving the processed images back to disk.
 *
 * All core processing stays within the GPU to maximize performance.
 */

#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/ImageBatch.hpp>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

class Timer {
public:
    Timer(const std::string& tag) : m_tag(tag), m_start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - m_start;
        std::cout << m_tag << ": " << diff.count() << " ms" << std::endl;
    }
private:
    std::string m_tag;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

int main(int argc, char* argv[]) {
    // --- Configuration ---
    const int target_width = 224;
    const int target_height = 224;
    const int kernel_size = 5;
    const float sigma = 1.0f;
    
    // 示例使用单张或多张图片路径
    std::vector<std::string> input_files = {"images/tabby_tiger_cat.jpg"}; 
    if (argc > 1) {
        input_files.clear();
        for (int i = 1; i < argc; ++i) input_files.push_back(argv[i]);
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    try {
        // 1. Load images (Using OpenCV for simplicity in C++ setup, then upload to GPU)
        std::vector<nvcv::Tensor> input_tensors;
        {
            Timer t("Load and Upload Images");
            for (const auto& path : input_files) {
                cv::Mat img = cv::imread(path);
                if (img.empty()) {
                    std::cerr << "Failed to load: " << path << std::endl;
                    continue;
                }
                
                // 分配显存并上传
                size_t img_bytes = img.total() * img.elemSize();
                uint8_t* d_img_ptr;
                CHECK_CUDA(cudaMalloc(&d_img_ptr, img_bytes));
                CHECK_CUDA(cudaMemcpyAsync(d_img_ptr, img.data, img_bytes, cudaMemcpyHostToDevice, stream));

                // 包装为 CV-CUDA Tensor (NHWC)
                input_tensors.emplace_back(nvcv::Tensor({{1, img.rows, img.cols, 3}, "NHWC"}, nvcv::TYPE_U8));
                auto data = input_tensors.back().exportData<nvcv::TensorDataStridedCuda>();
                CHECK_CUDA(cudaMemcpy2DAsync(data->basePtr(), data->stride(1), d_img_ptr, img.cols * 3, img.cols * 3, img.rows, cudaMemcpyDeviceToDevice, stream));
                
                CHECK_CUDA(cudaFree(d_img_ptr)); // 临时 Buffer 可释放
            }
        }

        if (input_tensors.empty()) return -1;

        // 2. Resize each image
        std::vector<nvcv::Tensor> resized_tensors;
        cvcuda::Resize resize_op;
        {
            Timer t("Resize Images");
            for (auto& in_tensor : input_tensors) {
                resized_tensors.emplace_back(nvcv::Tensor({{1, target_height, target_width, 3}, "NHWC"}, nvcv::TYPE_U8));
                resize_op(stream, in_tensor, resized_tensors.back(), NVCV_INTERP_LINEAR);
            }
        }

        // 3. Batch images into a single Tensor
        // 在 C++ 中，Batch 通常通过构造一个新的 Tensor 并拷贝数据实现
        int batch_size = resized_tensors.size();
        nvcv::Tensor batch_tensor({{batch_size, target_height, target_width, 3}, "NHWC"}, nvcv::TYPE_U8);
        {
            Timer t("Batching (Stack)");
            auto batch_data = batch_tensor.exportData<nvcv::TensorDataStridedCuda>();
            size_t single_img_bytes = target_height * target_width * 3;
            for (int i = 0; i < batch_size; ++i) {
                auto src_data = resized_tensors[i].exportData<nvcv::TensorDataStridedCuda>();
                CHECK_CUDA(cudaMemcpyAsync(
                    (uint8_t*)batch_data->basePtr() + i * single_img_bytes,
                    src_data->basePtr(),
                    single_img_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
            }
        }

        // 4. Apply Gaussian Blur on the batch
        nvcv::Tensor blurred_batch({{batch_size, target_height, target_width, 3}, "NHWC"}, nvcv::TYPE_U8);
        cvcuda::Gaussian blur_op({kernel_size, kernel_size}, batch_size);
        {
            Timer t("Gaussian Blur");
            blur_op(stream, batch_tensor, blurred_batch, {kernel_size, kernel_size}, {sigma, sigma}, NVCV_BORDER_CONSTANT);
        }

        // 5. Save images back to disk
        {
            Timer t("Download and Save");
            auto out_data = blurred_batch.exportData<nvcv::TensorDataStridedCuda>();
            size_t single_img_bytes = target_height * target_width * 3;
            std::vector<uint8_t> h_buf(single_img_bytes);

            for (int i = 0; i < batch_size; ++i) {
                CHECK_CUDA(cudaMemcpyAsync(h_buf.data(), (uint8_t*)out_data->basePtr() + i * single_img_bytes, single_img_bytes, cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));

                cv::Mat out_img(target_height, target_width, CV_8UC3, h_buf.data());
                std::string out_name = "output_" + std::to_string(i) + ".jpg";
                cv::imwrite(out_name, out_img);
                std::cout << "Saved: " << out_name << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
