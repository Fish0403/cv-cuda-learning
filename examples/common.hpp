#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

inline void CheckCuda(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + file + ":"
                                 + std::to_string(line));
    }
}

#define CUDA_CHECK(call) CheckCuda((call), __FILE__, __LINE__)

class Timer
{
public:
    explicit Timer(std::string tag)
        : m_tag(std::move(tag))
        , m_start(std::chrono::high_resolution_clock::now())
    {
    }

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms  = std::chrono::duration<double, std::milli>(end - m_start).count();
        std::cout << m_tag << ": " << ms << " ms" << std::endl;
    }

private:
    std::string m_tag;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

struct ImageDiffStats
{
    int mismatches  = 0;
    int maxAbsDiff  = 0;
    bool identical  = true;
};

inline ImageDiffStats CompareImagesU8(const cv::Mat &a, const cv::Mat &b)
{
    if (a.empty() || b.empty()) {
        throw std::runtime_error("CompareImagesU8: input image is empty.");
    }
    if (a.size() != b.size() || a.type() != b.type()) {
        throw std::runtime_error("CompareImagesU8: image size/type mismatch.");
    }
    if (a.depth() != CV_8U) {
        throw std::runtime_error("CompareImagesU8: only CV_8U images are supported.");
    }

    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::Mat diff1c = diff.reshape(1);

    double maxVal = 0.0;
    cv::minMaxLoc(diff1c, nullptr, &maxVal);

    ImageDiffStats stats;
    stats.mismatches = cv::countNonZero(diff1c);
    stats.maxAbsDiff = static_cast<int>(maxVal);
    stats.identical  = (stats.mismatches == 0);
    return stats;
}

inline void PrintCompareResult(const std::string &tag, const ImageDiffStats &stats)
{
    std::cout << tag << ": " << (stats.identical ? "IDENTICAL" : "DIFFERENT")
              << ", mismatches=" << stats.mismatches
              << ", max_abs_diff=" << stats.maxAbsDiff << "\n";
}

inline void PrintSectionHeader(int index, const std::string &title)
{
    std::cout << "\n========================================\n";
    std::cout << "【" << index << ". " << title << "】\n";
    std::cout << "========================================\n";
}
