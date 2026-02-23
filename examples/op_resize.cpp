// op_resize.cpp
// OpenCV vs CV-CUDA preprocessing demo:
// batch resize with simple timing blocks.

#include <cvcuda/OpResize.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "common.hpp"

int main()
{
    // Configuration
    const int batch = 1;
    const int src_w = 5120;
    const int src_h = 5120;
    const int dst_w = 4480;
    const int dst_h = 4480;
    const int warmup = 3;

    // Prepare host-side input batch
    std::vector<cv::Mat> cpu_srcs(batch);
    for (int i = 0; i < batch; ++i) {
        cpu_srcs[i] = cv::Mat(src_h, src_w, CV_8UC3, cv::Scalar(100, 100, 100));
    }
    std::vector<cv::Mat> cpu_dsts(batch);

    std::cout << "Batch: " << batch << " (" << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h << ")"
              << std::endl;

    PrintSectionHeader(1, "OpenCV CPU");
    {
        std::vector<cv::Mat> cpuWarmupDsts(batch);
        for (int w = 0; w < warmup; ++w) {
            for (int i = 0; i < batch; ++i) {
                cv::resize(cpu_srcs[i], cpuWarmupDsts[i], cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
            }
        }

        Timer t("OpenCV CPU Benchmark (Resize)");
        for (int i = 0; i < batch; ++i) {
            cv::resize(cpu_srcs[i], cpu_dsts[i], cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
        }
    }

    PrintSectionHeader(2, "OpenCV CUDA");
    bool hasOpenCvCuda = false;
    std::vector<cv::Mat> opencvCudaDsts(batch);
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            std::cout << "OpenCV CUDA skipped: no CUDA device found by OpenCV.\n";
        } else {
            hasOpenCvCuda = true;
            cv::cuda::setDevice(0);
            cv::cuda::Stream cvStream;
            std::vector<cv::cuda::GpuMat> gpuSrcs(batch), gpuResized(batch);

            {
                Timer t("OpenCV CUDA H2D upload(batch)");
                for (int i = 0; i < batch; ++i) {
                    gpuSrcs[i].upload(cpu_srcs[i], cvStream);
                }
                cvStream.waitForCompletion();
            }

            for (int w = 0; w < warmup; ++w) {
                for (int i = 0; i < batch; ++i) {
                    cv::cuda::resize(gpuSrcs[i], gpuResized[i], cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR,
                                     cvStream);
                }
            }
            cvStream.waitForCompletion();

            {
                Timer t("OpenCV CUDA Benchmark (Resize)");
                for (int i = 0; i < batch; ++i) {
                    cv::cuda::resize(gpuSrcs[i], gpuResized[i], cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR,
                                     cvStream);
                }
                cvStream.waitForCompletion();
            }

            {
                Timer t("OpenCV CUDA D2H download(batch)");
                for (int i = 0; i < batch; ++i) {
                    gpuResized[i].download(opencvCudaDsts[i], cvStream);
                }
                cvStream.waitForCompletion();
            }
        }
    } catch (const cv::Exception &e) {
        std::cout << "OpenCV CUDA skipped: " << e.what() << "\n";
    }

    PrintSectionHeader(3, "CV-CUDA");
    cudaStream_t stream = nullptr;
    try {
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Tensor layout is NHWC for both input and output.
        nvcv::Tensor gpu_src({{batch, src_h, src_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_dst({{batch, dst_h, dst_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        auto inDataOpt  = gpu_src.exportData<nvcv::TensorDataStridedCuda>();
        auto outDataOpt = gpu_dst.exportData<nvcv::TensorDataStridedCuda>();
        if (!inDataOpt || !outDataOpt) {
            throw std::runtime_error("Failed to export tensor data as CUDA strided buffers.");
        }
        const int64_t inSampleStride  = inDataOpt->stride(0);
        const int64_t inRowStride     = inDataOpt->stride(1);
        const int64_t outSampleStride = outDataOpt->stride(0);
        const int64_t outRowStride    = outDataOpt->stride(1);
        auto *inBase                  = inDataOpt->basePtr();
        auto *outBase                 = outDataOpt->basePtr();

        {
            Timer t("CV-CUDA H2D upload(batch)");
            for (int i = 0; i < batch; ++i) {
                CUDA_CHECK(cudaMemcpy2DAsync(inBase + i * inSampleStride, inRowStride, cpu_srcs[i].data, cpu_srcs[i].step,
                                             static_cast<size_t>(src_w * 3), src_h, cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cvcuda::Resize resizeOp;

        for (int w = 0; w < warmup; ++w) {
            resizeOp(stream, gpu_src, gpu_dst, NVCV_INTERP_LINEAR);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            Timer t("CV-CUDA Benchmark (Resize)");
            resizeOp(stream, gpu_src, gpu_dst, NVCV_INTERP_LINEAR);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::vector<cv::Mat> cvcudaDsts(batch);
        for (int i = 0; i < batch; ++i) {
            cvcudaDsts[i] = cv::Mat(dst_h, dst_w, CV_8UC3);
        }
        {
            Timer t("CV-CUDA D2H download(batch)");
            for (int i = 0; i < batch; ++i) {
                CUDA_CHECK(cudaMemcpy2DAsync(cvcudaDsts[i].data, cvcudaDsts[i].step, outBase + i * outSampleStride,
                                             outRowStride, static_cast<size_t>(dst_w * 3), dst_h,
                                             cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        PrintSectionHeader(4, "Compare");
        for (int i = 0; i < batch; ++i) {
            std::string suffix = (batch == 1) ? "" : (" [idx=" + std::to_string(i) + "]");
            if (hasOpenCvCuda) {
                PrintCompareResult("Compare OpenCV CPU vs OpenCV CUDA" + suffix, CompareImagesU8(cpu_dsts[i], opencvCudaDsts[i]));
            }
            PrintCompareResult("Compare OpenCV CPU vs CV-CUDA" + suffix, CompareImagesU8(cpu_dsts[i], cvcudaDsts[i]));
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        return 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
