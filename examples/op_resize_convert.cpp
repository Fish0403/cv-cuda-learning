// op_resize_convert.cpp
// OpenCV vs CV-CUDA preprocessing demo:
// batch resize + convertTo with simple timing blocks.

#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpResize.hpp>

#include "common.hpp"

int main()
{
    // Configuration
    const int batch = 25;
    const int src_w = 300;
    const int src_h = 300;
    const int dst_w = 224;
    const int dst_h = 224;
    const float alpha = 1.0f;   // scale factor
    const float beta = -127.5f; // bias

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Prepare host-side input batch
    std::vector<cv::Mat> cpu_srcs(batch);
    for (int i = 0; i < batch; ++i) {
        cpu_srcs[i] = cv::Mat(src_h, src_w, CV_8UC3, cv::Scalar(100, 100, 100));
    }

    std::cout << "Batch: " << batch << " (" << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h << ")" << std::endl;

    // ========================
    // OpenCV section
    // ========================
    {
        Timer t("OpenCV CPU (Resize + ConvertTo)");
        std::vector<cv::Mat> cpu_dsts(batch);
        for (int i = 0; i < batch; ++i) {
            cv::Mat resized;
            cv::resize(cpu_srcs[i], resized, cv::Size(dst_w, dst_h));
            resized.convertTo(cpu_dsts[i], CV_32FC3, alpha, beta);
        }
    }

    // ========================
    // CV-CUDA section
    // ========================
    try {
        // Tensor layout is NHWC for both input and output.
        nvcv::Tensor gpu_src({{batch, src_h, src_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_tmp({{batch, dst_h, dst_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_dst({{batch, dst_h, dst_w, 3}, "NHWC"}, nvcv::TYPE_F32);

        cvcuda::Resize resizeOp;
        cvcuda::ConvertTo convertOp;

        // Warmup to avoid first-run overhead in timing.
        resizeOp(stream, gpu_src, gpu_tmp, NVCV_INTERP_LINEAR);
        convertOp(stream, gpu_tmp, gpu_dst, alpha, beta);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            Timer t("CV-CUDA GPU (Batch Resize + ConvertTo)");
            resizeOp(stream, gpu_src, gpu_tmp, NVCV_INTERP_LINEAR);
            convertOp(stream, gpu_tmp, gpu_dst, alpha, beta);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
