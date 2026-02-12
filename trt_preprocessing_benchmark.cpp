/*
 * trt_preprocessing_benchmark.cpp
 *
 * Typical inference-oriented preprocessing benchmark:
 * 1) prepare a large input image and ROI list,
 * 2) run CPU baseline preprocessing (crop + convert + split),
 * 3) run GPU preprocessing with CV-CUDA fused operators,
 * 4) optionally feed the preprocessed tensor into TensorRT.
 *
 * Goal: compare end-to-end preprocessing latency for practical inference pipelines.
 */

#include <cstdint>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <NvInfer.h>

#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main() {
    // --- Model and I/O configuration ---
    const char* input_name  = "data";      // Model input tensor name.
    const char* output_name = "prob";      // Model output tensor name.
    const char* engine_path = "model.engine"; // TensorRT engine file path.

    // Image/grid configuration.
    const int width = 224 * 20;
    const int height = 224 * 20;
    const int crop_size = 224;
    const int batch_size = 25;
    const int channels = 3;
    const int num_classes = 2;

    int num_x = width / crop_size;
    int num_y = height / crop_size;
    int total_patches = num_x * num_y;
    int total_batches = (total_patches + batch_size - 1) / batch_size;

    size_t img_size = (size_t)width * height * channels;
    std::cout << "Image: " << width << "x" << height << " (" << total_patches << " patches)" << std::endl;

    uint8_t* h_pinned_input;
    CHECK_CUDA(cudaHostAlloc(&h_pinned_input, img_size, cudaHostAllocDefault));
    memset(h_pinned_input, 128, img_size);

    std::vector<NVCVRectI> all_rois;
    for (int y = 0; y < num_y; ++y) {
        for (int x = 0; x < num_x; ++x) {
            all_rois.push_back({x * crop_size, y * crop_size, crop_size, crop_size});
        }
    }

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to find engine file: " << engine_path << std::endl;
        return -1;
    }
    file.seekg(0, std::ios::end);
    size_t engine_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * num_classes * sizeof(float)));

    // --- Method A: CPU preprocessing baseline ---
    {
        std::cout << "Method A (CPU Pre - OpenCV Opt): ";
        static cudaStream_t stream = nullptr;
        if (!stream) CHECK_CUDA(cudaStreamCreate(&stream));
        static float* h_pinned_batch = nullptr;
        if (!h_pinned_batch) CHECK_CUDA(cudaHostAlloc(&h_pinned_batch, batch_size * 3 * crop_size * crop_size * sizeof(float), cudaHostAllocDefault));
        static float* d_trt_input = nullptr;
        if (!d_trt_input) CHECK_CUDA(cudaMalloc(&d_trt_input, batch_size * 3 * crop_size * crop_size * sizeof(float)));

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat cpu_img(height, width, CV_8UC3, h_pinned_input);
        for (int b = 0; b < total_batches; ++b) {
            int current_batch_size = std::min(batch_size, total_patches - b * batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                NVCVRectI r = all_rois[b * batch_size + i];
                cv::Mat crop = cpu_img(cv::Rect(r.x, r.y, crop_size, crop_size));
                cv::Mat float_crop;
                crop.convertTo(float_crop, CV_32FC3, 1.0/255.0);
                float* base = h_pinned_batch + i * (3 * crop_size * crop_size);
                std::vector<cv::Mat> planes = {
                    cv::Mat(crop_size, crop_size, CV_32FC1, base),
                    cv::Mat(crop_size, crop_size, CV_32FC1, base + crop_size * crop_size),
                    cv::Mat(crop_size, crop_size, CV_32FC1, base + 2 * crop_size * crop_size)
                };
                cv::split(float_crop, planes);
            }
            CHECK_CUDA(cudaMemcpyAsync(d_trt_input, h_pinned_batch, batch_size * 3 * crop_size * crop_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    }

    // --- Method B: GPU preprocessing (CV-CUDA optimized fused batch) ---
    {
        std::cout << "Method B (GPU Pre - Optimized Fused): ";
        static cudaStream_t stream = nullptr;
        if (!stream) CHECK_CUDA(cudaStreamCreate(&stream));
        static uint8_t* d_full_img = nullptr;
        if (!d_full_img) CHECK_CUDA(cudaMalloc(&d_full_img, img_size));
        
        static uint8_t* d_batch_u8 = nullptr;
        if (!d_batch_u8) CHECK_CUDA(cudaMalloc(&d_batch_u8, batch_size * crop_size * crop_size * 3));

        static float* d_trt_input_nchw = nullptr;
        if (!d_trt_input_nchw) CHECK_CUDA(cudaMalloc(&d_trt_input_nchw, batch_size * 3 * crop_size * crop_size * sizeof(float)));

        static cvcuda::ResizeCropConvertReformat fused_op;

        // Create tensors once to avoid per-iteration object construction.
        nvcv::Tensor batch_u8_tensor({{batch_size, crop_size, crop_size, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor batch_f32_tensor({{batch_size, 3, crop_size, crop_size}, "NCHW"}, nvcv::TYPE_F32);
        
        auto u8_data = batch_u8_tensor.exportData<nvcv::TensorDataStridedCuda>();
        auto f32_data = batch_f32_tensor.exportData<nvcv::TensorDataStridedCuda>();

        // Warmup.
        CHECK_CUDA(cudaMemcpyAsync(d_full_img, h_pinned_input, img_size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        auto start = std::chrono::high_resolution_clock::now();
        
        // 1) Upload full image.
        CHECK_CUDA(cudaMemcpyAsync(d_full_img, h_pinned_input, img_size, cudaMemcpyHostToDevice, stream));

        for (int b = 0; b < total_batches; ++b) {
            int current_batch_size = std::min(batch_size, total_patches - b * batch_size);
            
            // 2) Batch ROI gathering into contiguous device memory (D2D gather).
            for (int i = 0; i < current_batch_size; ++i) {
                NVCVRectI r = all_rois[b * batch_size + i];
                CHECK_CUDA(cudaMemcpy2DAsync(
                    d_batch_u8 + i * crop_size * crop_size * 3, crop_size * 3,
                    d_full_img + r.y * width * 3 + r.x * 3, width * 3,
                    crop_size * 3, crop_size,
                    cudaMemcpyDeviceToDevice, stream
                ));
            }

            // 3) Copy gathered ROI buffer into tensor memory.
            CHECK_CUDA(cudaMemcpyAsync(u8_data->basePtr(), d_batch_u8, current_batch_size * crop_size * crop_size * 3, cudaMemcpyDeviceToDevice, stream));

            // 4) Run fused preprocessing operator once per batch (outside ROI loop).
            fused_op(stream, batch_u8_tensor, batch_f32_tensor, 
                     {crop_size, crop_size}, NVCV_INTERP_LINEAR,
                     {0, 0}, 
                     NVCV_CHANNEL_REVERSE, 
                     1.0f/255.0f, 0.0f, false);
            
            // 5) Copy preprocessed output to inference input buffer.
            CHECK_CUDA(cudaMemcpyAsync(d_trt_input_nchw, f32_data->basePtr(), current_batch_size * 3 * crop_size * crop_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

        // Inference sanity check.
        context->setInputShape(input_name, nvinfer1::Dims4{batch_size, 3, crop_size, crop_size});
        context->setTensorAddress(input_name, d_trt_input_nchw);
        context->setTensorAddress(output_name, d_output);
        context->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    delete context; delete engine; delete runtime;
    CHECK_CUDA(cudaFree(d_output)); CHECK_CUDA(cudaFreeHost(h_pinned_input));
    return 0;
}
