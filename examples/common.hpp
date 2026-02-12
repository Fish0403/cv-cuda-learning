#pragma once

#include <chrono>
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
