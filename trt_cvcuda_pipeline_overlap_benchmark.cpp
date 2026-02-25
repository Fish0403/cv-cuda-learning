/*
 * trt_cvcuda_pipeline_overlap_benchmark.cpp
 *
 * Purpose:
 * 1) Compare two production-style pipeline patterns:
 *    - Serial: Upload -> CV-CUDA preprocess -> TensorRT inference (single stream)
 *    - Overlap: Upload and (CV-CUDA preprocess + TensorRT inference) overlapped across batches
 *      using dual streams, ping-pong buffers, and CUDA events.
 * 2) Print total / avg / speedup to quantify pipeline-overlap gains.
 *
 * Notes:
 * - Preprocess path uses CV-CUDA ResizeCropConvertReformat (fused op).
 * - Both modes run warmup iterations to reduce cold-start noise.
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                                  \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                      << cudaGetErrorString(err__) << std::endl;                           \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

int main() {
    const char *engine_path = "model_fp16.engine";
    const char *input_name = "data";
    const char *output_name = "prob";

    const int batch_size = 25;
    const int src_h = 640;
    const int src_w = 640;
    const int dst_h = 224;
    const int dst_w = 224;
    const int channels = 3;
    const int num_classes = 2;
    const int total_batches = 120;
    const int warmup_iters = 20;

    const size_t h2d_bytes = static_cast<size_t>(batch_size) * src_h * src_w * channels * sizeof(uint8_t);
    const size_t out_bytes = static_cast<size_t>(batch_size) * num_classes * sizeof(float);

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine: " << engine_path << std::endl;
        return -1;
    }
    file.seekg(0, std::ios::end);
    const size_t engine_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);

    Logger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "createInferRuntime failed" << std::endl;
        return -1;
    }

    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!engine) {
        std::cerr << "deserializeCudaEngine failed" << std::endl;
        delete runtime;
        return -1;
    }

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "createExecutionContext failed" << std::endl;
        delete engine;
        delete runtime;
        return -1;
    }

    const nvinfer1::Dims4 input_shape{batch_size, channels, dst_h, dst_w};
    if (!context->setInputShape(input_name, input_shape)) {
        std::cerr << "setInputShape failed for " << input_name << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    // Ping-pong buffers (slot 0/1) for cross-batch overlap.
    uint8_t *h_u8[2] = {nullptr, nullptr};
    uint8_t *d_u8[2] = {nullptr, nullptr};
    float *d_out[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaHostAlloc(&h_u8[i], h2d_bytes, cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&d_u8[i], h2d_bytes));
        CHECK_CUDA(cudaMalloc(&d_out[i], out_bytes));
        std::memset(h_u8[i], 0, h2d_bytes);
    }

    // Duplicate CV-CUDA input/output tensors to avoid read-write conflicts.
    nvcv::Tensor u8_tensor[2] = {
        nvcv::Tensor({{batch_size, src_h, src_w, channels}, "NHWC"}, nvcv::TYPE_U8),
        nvcv::Tensor({{batch_size, src_h, src_w, channels}, "NHWC"}, nvcv::TYPE_U8)
    };
    nvcv::Tensor f32_tensor[2] = {
        nvcv::Tensor({{batch_size, channels, dst_h, dst_w}, "NCHW"}, nvcv::TYPE_F32),
        nvcv::Tensor({{batch_size, channels, dst_h, dst_w}, "NCHW"}, nvcv::TYPE_F32)
    };

    auto u8_data0 = u8_tensor[0].exportData<nvcv::TensorDataStridedCuda>();
    auto u8_data1 = u8_tensor[1].exportData<nvcv::TensorDataStridedCuda>();
    auto f32_data0 = f32_tensor[0].exportData<nvcv::TensorDataStridedCuda>();
    auto f32_data1 = f32_tensor[1].exportData<nvcv::TensorDataStridedCuda>();

    nvcv::Byte *u8_ptr[2] = {u8_data0->basePtr(), u8_data1->basePtr()};
    float *f32_ptr[2] = {reinterpret_cast<float *>(f32_data0->basePtr()),
                         reinterpret_cast<float *>(f32_data1->basePtr())};

    cvcuda::ResizeCropConvertReformat fused_op;

    cudaStream_t serial_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t proc_stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&serial_stream));
    CHECK_CUDA(cudaStreamCreate(&copy_stream));
    CHECK_CUDA(cudaStreamCreate(&proc_stream));

    cudaEvent_t copy_done[2];
    CHECK_CUDA(cudaEventCreateWithFlags(&copy_done[0], cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&copy_done[1], cudaEventDisableTiming));

    // Prefill host data to avoid CPU data generation becoming a benchmark bottleneck.
    for (int s = 0; s < 2; ++s) {
        const uint8_t v = static_cast<uint8_t>(17 + s * 13);
        std::memset(h_u8[s], v, h2d_bytes);
    }

    // ------------------ Mode A: serial pipeline (upload -> preprocess -> infer) ------------------
    for (int i = 0; i < warmup_iters; ++i) {
        const int slot = i & 1;
        CHECK_CUDA(cudaMemcpyAsync(d_u8[slot], h_u8[slot], h2d_bytes, cudaMemcpyHostToDevice, serial_stream));
        CHECK_CUDA(cudaMemcpyAsync(u8_ptr[slot], d_u8[slot], h2d_bytes, cudaMemcpyDeviceToDevice, serial_stream));

        fused_op(serial_stream, u8_tensor[slot], f32_tensor[slot],
                 {dst_h, dst_w}, NVCV_INTERP_LINEAR, {0, 0},
                 NVCV_CHANNEL_REVERSE, 1.0f / 255.0f, 0.0f, false);

        context->setTensorAddress(input_name, f32_ptr[slot]);
        context->setTensorAddress(output_name, d_out[slot]);
        if (!context->enqueueV3(serial_stream)) {
            std::cerr << "Serial warmup enqueueV3 failed" << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(serial_stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < total_batches; ++b) {
        const int slot = b & 1;

        CHECK_CUDA(cudaMemcpyAsync(d_u8[slot], h_u8[slot], h2d_bytes, cudaMemcpyHostToDevice, serial_stream));
        CHECK_CUDA(cudaMemcpyAsync(u8_ptr[slot], d_u8[slot], h2d_bytes, cudaMemcpyDeviceToDevice, serial_stream));

        fused_op(serial_stream, u8_tensor[slot], f32_tensor[slot],
                 {dst_h, dst_w}, NVCV_INTERP_LINEAR, {0, 0},
                 NVCV_CHANNEL_REVERSE, 1.0f / 255.0f, 0.0f, false);

        context->setTensorAddress(input_name, f32_ptr[slot]);
        context->setTensorAddress(output_name, d_out[slot]);
        if (!context->enqueueV3(serial_stream)) {
            std::cerr << "Serial enqueueV3 failed at batch " << b << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(serial_stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    const double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double serial_avg = serial_ms / total_batches;

    // ------------------ Mode B: overlapped pipeline (upload || preprocess+infer) ------------------
    for (int i = 0; i < warmup_iters; ++i) {
        const int slot = i & 1;
        CHECK_CUDA(cudaMemcpyAsync(d_u8[slot], h_u8[slot], h2d_bytes, cudaMemcpyHostToDevice, copy_stream));
        // Record an "upload done" event on copy_stream (non-blocking to CPU).
        CHECK_CUDA(cudaEventRecord(copy_done[slot], copy_stream));
        // Make proc_stream wait for that event before preprocess/infer.
        CHECK_CUDA(cudaStreamWaitEvent(proc_stream, copy_done[slot], 0));

        CHECK_CUDA(cudaMemcpyAsync(u8_ptr[slot], d_u8[slot], h2d_bytes, cudaMemcpyDeviceToDevice, proc_stream));
        fused_op(proc_stream, u8_tensor[slot], f32_tensor[slot],
                 {dst_h, dst_w}, NVCV_INTERP_LINEAR, {0, 0},
                 NVCV_CHANNEL_REVERSE, 1.0f / 255.0f, 0.0f, false);

        context->setTensorAddress(input_name, f32_ptr[slot]);
        context->setTensorAddress(output_name, d_out[slot]);
        if (!context->enqueueV3(proc_stream)) {
            std::cerr << "Overlap warmup enqueueV3 failed" << std::endl;
            return -1;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(copy_stream));
    CHECK_CUDA(cudaStreamSynchronize(proc_stream));

    auto t2 = std::chrono::high_resolution_clock::now();

    // Submit batch-0 upload first, then overlap "next upload" with "current preprocess+infer".
    CHECK_CUDA(cudaMemcpyAsync(d_u8[0], h_u8[0], h2d_bytes, cudaMemcpyHostToDevice, copy_stream));
    // Record batch-0 upload completion for proc_stream dependency.
    CHECK_CUDA(cudaEventRecord(copy_done[0], copy_stream));

    for (int b = 0; b < total_batches; ++b) {
        const int cur = b & 1;
        const int next = (b + 1) & 1;

        if (b + 1 < total_batches) {
            CHECK_CUDA(cudaMemcpyAsync(d_u8[next], h_u8[next], h2d_bytes, cudaMemcpyHostToDevice, copy_stream));
            // Record next upload completion; it becomes cur in the next iteration.
            CHECK_CUDA(cudaEventRecord(copy_done[next], copy_stream));
        }

        // Wait until current batch upload is done, then start preprocess+infer.
        CHECK_CUDA(cudaStreamWaitEvent(proc_stream, copy_done[cur], 0));

        CHECK_CUDA(cudaMemcpyAsync(u8_ptr[cur], d_u8[cur], h2d_bytes, cudaMemcpyDeviceToDevice, proc_stream));
        fused_op(proc_stream, u8_tensor[cur], f32_tensor[cur],
                 {dst_h, dst_w}, NVCV_INTERP_LINEAR, {0, 0},
                 NVCV_CHANNEL_REVERSE, 1.0f / 255.0f, 0.0f, false);

        context->setTensorAddress(input_name, f32_ptr[cur]);
        context->setTensorAddress(output_name, d_out[cur]);
        if (!context->enqueueV3(proc_stream)) {
            std::cerr << "Overlap enqueueV3 failed at batch " << b << std::endl;
            return -1;
        }
    }

    CHECK_CUDA(cudaStreamSynchronize(copy_stream));
    CHECK_CUDA(cudaStreamSynchronize(proc_stream));
    auto t3 = std::chrono::high_resolution_clock::now();

    const double overlap_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double overlap_avg = overlap_ms / total_batches;
    const double speedup = serial_ms / std::max(overlap_ms, 1e-6);

    std::cout << "\n=== Upload vs (CV-CUDA preprocess + TRT inference) ===" << std::endl;
    std::cout << "Batches: " << total_batches << ", batch_size: " << batch_size
              << ", src=" << src_h << "x" << src_w
              << ", dst=" << dst_h << "x" << dst_w << std::endl;
    std::cout << "Serial   : total " << serial_ms << " ms, avg " << serial_avg << " ms/batch" << std::endl;
    std::cout << "Overlap  : total " << overlap_ms << " ms, avg " << overlap_avg << " ms/batch" << std::endl;
    std::cout << "Speedup  : x" << speedup << std::endl;

    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaFree(d_out[i]));
        CHECK_CUDA(cudaFree(d_u8[i]));
        CHECK_CUDA(cudaFreeHost(h_u8[i]));
    }
    CHECK_CUDA(cudaEventDestroy(copy_done[0]));
    CHECK_CUDA(cudaEventDestroy(copy_done[1]));
    CHECK_CUDA(cudaStreamDestroy(serial_stream));
    CHECK_CUDA(cudaStreamDestroy(copy_stream));
    CHECK_CUDA(cudaStreamDestroy(proc_stream));

    delete context;
    delete engine;
    delete runtime;

    return 0;
}
