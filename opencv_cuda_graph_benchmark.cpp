/*
 * opencv_cuda_graph_benchmark.cpp
 *
 * Operator-only benchmark for three preprocess paths:
 * 1) OpenCV CUDA Non-Graph: per-iteration resize/convertTo/split launches
 * 2) OpenCV CUDA Graph: capture once, then launch via cudaGraphLaunch
 * 3) CV-CUDA Fused: ResizeCropConvertReformat only
 */

#include <cuda_runtime.h>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                                  \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                      << cudaGetErrorString(err__) << std::endl;                           \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

int main() {
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
        std::cout << "No OpenCV CUDA device found." << std::endl;
        return 0;
    }

    cv::cuda::setDevice(0);

    const int width = 5120;
    const int height = 5120;
    const int channels = 3;
    const int crop_size = 224;
    const int batch_size = 25;   // 5x5 ROIs
    const int warmup_iters = 20;
    const int bench_iters = 200;

    const size_t img_bytes = static_cast<size_t>(width) * height * channels;
    const size_t nchw_bytes = static_cast<size_t>(batch_size) * 3 * crop_size * crop_size * sizeof(float);

    uint8_t *h_img = nullptr;
    uint8_t *d_img = nullptr;
    float *d_nchw = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_img, img_bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d_img, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_nchw, nchw_bytes));
    std::memset(h_img, 127, img_bytes);

    cv::cuda::Stream cv_stream;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream);

    CHECK_CUDA(cudaMemcpyAsync(d_img, h_img, img_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cv::cuda::GpuMat full_img(height, width, CV_8UC3, d_img, static_cast<size_t>(width * 3));

    const int grid_x = 5;
    const int grid_y = 5;
    const int roi_w = width / grid_x;
    const int roi_h = height / grid_y;
    std::vector<cv::cuda::GpuMat> rois;
    rois.reserve(batch_size);
    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            int rx = x * roi_w;
            int ry = y * roi_h;
            rois.emplace_back(full_img, cv::Rect(rx, ry, roi_w, roi_h));
        }
    }

    std::vector<cv::cuda::GpuMat> resized(batch_size);
    std::vector<cv::cuda::GpuMat> f32(batch_size);
    std::vector<std::vector<cv::cuda::GpuMat>> dst_channels(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        float *base = d_nchw + static_cast<size_t>(i) * 3 * crop_size * crop_size;
        cv::cuda::GpuMat ch0(crop_size, crop_size, CV_32FC1, base,
                             static_cast<size_t>(crop_size * sizeof(float)));
        cv::cuda::GpuMat ch1(crop_size, crop_size, CV_32FC1, base + crop_size * crop_size,
                             static_cast<size_t>(crop_size * sizeof(float)));
        cv::cuda::GpuMat ch2(crop_size, crop_size, CV_32FC1, base + 2 * crop_size * crop_size,
                             static_cast<size_t>(crop_size * sizeof(float)));
        dst_channels[i] = {ch0, ch1, ch2};
    }

    // OpenCV CUDA preprocess chain for one iteration:
    // resize -> convertTo(float, scale) -> split(CHW).
    auto run_opencv_once = [&]() {
        for (int i = 0; i < batch_size; ++i) {
            cv::cuda::resize(rois[i], resized[i], cv::Size(crop_size, crop_size), 0.0, 0.0, cv::INTER_LINEAR,
                             cv_stream);
            resized[i].convertTo(f32[i], CV_32FC3, 1.0 / 255.0, 0.0, cv_stream);
            cv::cuda::split(f32[i], dst_channels[i], cv_stream);
        }
    };

    cvcuda::ResizeCropConvertReformat fused_op;
    nvcv::Tensor batch_u8_tensor({{batch_size, roi_h, roi_w, 3}, "NHWC"}, nvcv::TYPE_U8);
    nvcv::Tensor batch_f32_tensor({{batch_size, 3, crop_size, crop_size}, "NCHW"}, nvcv::TYPE_F32);
    auto u8_data = batch_u8_tensor.exportData<nvcv::TensorDataStridedCuda>();
    if (!u8_data) {
        std::cerr << "Failed to export CV-CUDA input tensor data." << std::endl;
        CHECK_CUDA(cudaFree(d_nchw));
        CHECK_CUDA(cudaFree(d_img));
        CHECK_CUDA(cudaFreeHost(h_img));
        return -1;
    }
    CHECK_CUDA(cudaMemsetAsync(u8_data->basePtr(), 123, static_cast<size_t>(batch_size) * roi_h * roi_w * channels, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto run_cvcuda_once = [&]() -> int {
        fused_op(stream, batch_u8_tensor, batch_f32_tensor,
                 {crop_size, crop_size}, NVCV_INTERP_LINEAR,
                 {0, 0}, NVCV_CHANNEL_REVERSE, 1.0f / 255.0f, 0.0f, false);
        return 0;
    };

    // ===== Mode A: OpenCV CUDA Non-Graph =====
    // Submit preprocess operators per iteration (normal launch path).
    for (int i = 0; i < warmup_iters; ++i) {
        run_opencv_once();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        run_opencv_once();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    const double non_graph_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ===== Mode B: OpenCV CUDA Graph =====
    // Capture one stable preprocess iteration, then replay it by cudaGraphLaunch.
    for (int i = 0; i < warmup_iters; ++i) {
        run_opencv_once();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    // Step 1) Begin stream capture.
    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        std::cerr << "cudaStreamBeginCapture failed: " << cudaGetErrorString(st) << std::endl;
        CHECK_CUDA(cudaFree(d_nchw));
        CHECK_CUDA(cudaFree(d_img));
        CHECK_CUDA(cudaFreeHost(h_img));
        return -1;
    }

    // Step 2) Record one iteration into the graph.
    run_opencv_once();

    // Step 3) End capture and materialize graph object.
    st = cudaStreamEndCapture(stream, &graph);
    if (st != cudaSuccess || graph == nullptr) {
        std::cerr << "cudaStreamEndCapture failed: " << cudaGetErrorString(st) << std::endl;
        CHECK_CUDA(cudaFree(d_nchw));
        CHECK_CUDA(cudaFree(d_img));
        CHECK_CUDA(cudaFreeHost(h_img));
        return -1;
    }

    // Step 4) Instantiate executable graph.
    st = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if (st != cudaSuccess || graph_exec == nullptr) {
        std::cerr << "cudaGraphInstantiate failed: " << cudaGetErrorString(st) << std::endl;
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaFree(d_nchw));
        CHECK_CUDA(cudaFree(d_img));
        CHECK_CUDA(cudaFreeHost(h_img));
        return -1;
    }

    // Step 5) Replay graph for warmup.
    for (int i = 0; i < warmup_iters; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t2 = std::chrono::high_resolution_clock::now();
    // Step 6) Benchmark replay cost (one launch submits the whole captured chain).
    for (int i = 0; i < bench_iters; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t3 = std::chrono::high_resolution_clock::now();
    const double graph_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ===== Mode C: CV-CUDA Fused Operator =====
    // Use ResizeCropConvertReformat as fused preprocess baseline.
    for (int i = 0; i < warmup_iters; ++i) {
        if (run_cvcuda_once() != 0) return -1;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        if (run_cvcuda_once() != 0) return -1;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t5 = std::chrono::high_resolution_clock::now();
    const double cvcuda_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    const double non_graph_avg = non_graph_ms / bench_iters;
    const double graph_avg = graph_ms / bench_iters;
    const double cvcuda_avg = cvcuda_ms / bench_iters;
    const double graph_speedup = non_graph_ms / std::max(graph_ms, 1e-6);
    const double cvcuda_vs_non_graph = non_graph_ms / std::max(cvcuda_ms, 1e-6);
    const double cvcuda_vs_graph = graph_ms / std::max(cvcuda_ms, 1e-6);

    std::cout << "\n=== Operator-only preprocess benchmark (OpenCV CUDA vs CV-CUDA) ===" << std::endl;
    std::cout << "Image=" << width << "x" << height << ", batch=" << batch_size
              << ", crop=" << crop_size << "x" << crop_size
              << ", iters=" << bench_iters << std::endl;
    std::cout << "OpenCV Non-Graph: total " << non_graph_ms << " ms, avg " << non_graph_avg << " ms/iter" << std::endl;
    std::cout << "OpenCV Graph    : total " << graph_ms << " ms, avg " << graph_avg << " ms/iter" << std::endl;
    std::cout << "CV-CUDA Fused(op): total " << cvcuda_ms << " ms, avg " << cvcuda_avg << " ms/iter" << std::endl;
    std::cout << "Graph speedup vs Non-Graph  : x" << graph_speedup << std::endl;
    std::cout << "CV-CUDA speedup vs Non-Graph: x" << cvcuda_vs_non_graph << std::endl;
    std::cout << "CV-CUDA speedup vs Graph    : x" << cvcuda_vs_graph << std::endl;

    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_nchw));
    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFreeHost(h_img));

    return 0;
}
