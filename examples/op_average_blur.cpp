// op_average_blur.cpp
// OpenCV CPU vs CV-CUDA AverageBlur benchmark:
// - random grayscale input (5120x5120)
// - warmup + benchmark loops
// - H2D / kernel / D2H timing blocks
// - output consistency check (CPU vs GPU)
//
// Observation:
// - As kernel size increases (e.g., from 7x7 to 29x29), CV-CUDA AverageBlur
//   kernel time can grow significantly and may become slower than OpenCV CPU
//   for this workload.

#include <cvcuda/OpAverageBlur.hpp>

#include "common.hpp"

int main()
{
    // Configuration
    const int width = 5120;
    const int height = 5120;
    const int channels = 1;
    const int kernel = 29;
    const int warmup = 3;
    const int iters = 10;

    cv::Mat cpu_src(height, width, CV_8UC1);
    cv::randu(cpu_src, 0, 256);

    cv::Mat cpu_dst(height, width, CV_8UC1);

    std::cout << "Image: " << width << "x" << height << "x" << channels << "\n";
    std::cout << "Kernel: " << kernel << "x" << kernel << ", warmup=" << warmup << ", iters=" << iters << "\n";

    // ========================
    // OpenCV section
    // ========================
    try {
        {
            Timer t("OpenCV CPU Warmup (AverageBlur)");
            for (int i = 0; i < warmup; ++i) {
                cv::blur(cpu_src, cpu_dst, cv::Size(kernel, kernel), cv::Point(-1, -1), cv::BORDER_REPLICATE);
            }
        }
        {
            Timer t("OpenCV CPU Benchmark (AverageBlur)");
            for (int i = 0; i < iters; ++i) {
                cv::blur(cpu_src, cpu_dst, cv::Size(kernel, kernel), cv::Point(-1, -1), cv::BORDER_REPLICATE);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    }

    // ========================
    // CV-CUDA section
    // ========================
    cudaStream_t stream = nullptr;
    try {
        CUDA_CHECK(cudaStreamCreate(&stream));

        nvcv::Tensor gpu_in({{1, height, width, channels}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_out({{1, height, width, channels}, "NHWC"}, nvcv::TYPE_U8);

        auto in_data = gpu_in.exportData<nvcv::TensorDataStridedCuda>();
        auto out_data = gpu_out.exportData<nvcv::TensorDataStridedCuda>();
        if (!in_data || !out_data) {
            throw std::runtime_error("Failed to export tensor data as CUDA strided buffers.");
        }

        {
            Timer t("H2D upload(gray)");
            CUDA_CHECK(cudaMemcpy2DAsync(in_data->basePtr(), in_data->stride(1), cpu_src.data, cpu_src.step,
                                         static_cast<size_t>(width * channels), height, cudaMemcpyHostToDevice,
                                         stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cvcuda::AverageBlur avg_blur({kernel, kernel}, 1);

        {
            Timer t("CV-CUDA GPU Warmup (AverageBlur)");
            for (int i = 0; i < warmup; ++i) {
                avg_blur(stream, gpu_in, gpu_out, {kernel, kernel}, {-1, -1}, NVCV_BORDER_REPLICATE);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            Timer t("CV-CUDA GPU Benchmark (AverageBlur)");
            for (int i = 0; i < iters; ++i) {
                avg_blur(stream, gpu_in, gpu_out, {kernel, kernel}, {-1, -1}, NVCV_BORDER_REPLICATE);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cv::Mat gpu_dst(height, width, CV_8UC1);
        {
            Timer t("D2H download(gray)");
            CUDA_CHECK(cudaMemcpy2D(gpu_dst.data, gpu_dst.step, out_data->basePtr(), out_data->stride(1),
                                    static_cast<size_t>(width * channels), height, cudaMemcpyDeviceToHost));
        }

        int mismatches = 0;
        int max_abs_diff = 0;
        for (int y = 0; y < height; ++y) {
            const uchar *cpu_row = cpu_dst.ptr<uchar>(y);
            const uchar *gpu_row = gpu_dst.ptr<uchar>(y);
            for (int x = 0; x < width; ++x) {
                int diff = std::abs(static_cast<int>(cpu_row[x]) - static_cast<int>(gpu_row[x]));
                if (diff != 0) {
                    ++mismatches;
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                    }
                }
            }
        }

        std::cout << "Compare CPU vs GPU: "
                  << (mismatches == 0 ? "IDENTICAL" : "DIFFERENT")
                  << ", mismatches=" << mismatches
                  << ", max_abs_diff=" << max_abs_diff << "\n";

    } catch (const std::exception &e) {
        std::cerr << "CV-CUDA Error: " << e.what() << std::endl;
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        return 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
