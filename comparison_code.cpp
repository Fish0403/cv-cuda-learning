#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpConvertTo.hpp> // 对应 OpenCV 的 ConvertTo
#include <nvcv/Tensor.hpp>
#include <cuda_runtime.h>

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

int main() {
    // --- 配置 ---
    const int batch = 25;
    const int src_w = 300, src_h = 300;
    const int dst_w = 224, dst_h = 224;
    const float alpha = 1.0f;       // 缩放因子
    const float beta = -127.5f;     // 偏移量 (减去均值)

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. 准备 CPU 数据
    std::vector<cv::Mat> cpu_srcs(batch);
    for (int i = 0; i < batch; ++i) {
        cpu_srcs[i] = cv::Mat(src_h, src_w, CV_8UC3, cv::Scalar(100, 100, 100));
    }

    std::cout << "Testing Batch Size: " << batch << " (" << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h << ")\n" << std::endl;

    // --- OpenCV CPU 测试 ---
    {
        Timer t("OpenCV CPU (Resize + ConvertTo)");
        std::vector<cv::Mat> cpu_dsts(batch);
        for (int i = 0; i < batch; ++i) {
            cv::Mat resized;
            cv::resize(cpu_srcs[i], resized, cv::Size(dst_w, dst_h));
            resized.convertTo(cpu_dsts[i], CV_32FC3, alpha, beta);
        }
    }

    // --- CV-CUDA GPU 测试 ---
    try {
        // 分配内存: 输入是 Batch 的 Uint8, 输出是 Batch 的 Float32
        nvcv::Tensor gpu_src({{batch, src_h, src_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_tmp({{batch, dst_h, dst_w, 3}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor gpu_dst({{batch, dst_h, dst_w, 3}, "NHWC"}, nvcv::TYPE_F32);

        cvcuda::Resize resizeOp;
        cvcuda::ConvertTo convertOp;

        // 预热
        resizeOp(stream, gpu_src, gpu_tmp, NVCV_INTERP_LINEAR);
        convertOp(stream, gpu_tmp, gpu_dst, alpha, beta);
        cudaStreamSynchronize(stream);

        {
            Timer t("CV-CUDA GPU (Batch Resize + ConvertTo)");
            // 1. 批量缩放
            resizeOp(stream, gpu_src, gpu_tmp, NVCV_INTERP_LINEAR);
            // 2. 批量转换类型并减均值
            convertOp(stream, gpu_tmp, gpu_dst, alpha, beta);
            
            cudaStreamSynchronize(stream);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}