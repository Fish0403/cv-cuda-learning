// op_warp_affine.cpp
// OpenCV vs CV-CUDA affine rotation demo:
// load grayscale image, rotate by fixed 5 degrees, save CV-CUDA output image.

#include <cvcuda/OpWarpAffine.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "common.hpp"

static void BuildRotationXform(int width, int height, double angleDeg, NVCVAffineTransform xform)
{
    const cv::Point2f center(width * 0.5f, height * 0.5f);
    const cv::Mat rot2x3 = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    xform[0] = static_cast<float>(rot2x3.at<double>(0, 0));
    xform[1] = static_cast<float>(rot2x3.at<double>(0, 1));
    xform[2] = static_cast<float>(rot2x3.at<double>(0, 2));
    xform[3] = static_cast<float>(rot2x3.at<double>(1, 0));
    xform[4] = static_cast<float>(rot2x3.at<double>(1, 1));
    xform[5] = static_cast<float>(rot2x3.at<double>(1, 2));
}

int main(int argc, char **argv)
{
    const std::string imagePath = (argc > 1) ? argv[1] : "../images/CS.bmp";
    cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Failed to load grayscale image: " << imagePath << std::endl;
        return 1;
    }

    const int width = gray.cols;
    const int height = gray.rows;
    const int warmup = 3;
    std::cout << "Input(gray): " << width << "x" << height << std::endl;

    const double angleDeg = 5.0;
    const cv::Point2f center(width * 0.5f, height * 0.5f);
    const cv::Mat rot2x3 = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    PrintSectionHeader(1, "OpenCV CPU");
    cv::Mat cvWarped(height, width, CV_8UC1);
    for (int w = 0; w < warmup; ++w) {
        cv::warpAffine(gray, cvWarped, rot2x3, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    }
    {
        Timer t("OpenCV CPU Benchmark (WarpAffine)");
        cv::warpAffine(gray, cvWarped, rot2x3, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    }

    PrintSectionHeader(2, "OpenCV CUDA");
    cv::Mat cvCudaWarped(height, width, CV_8UC1);
    bool hasOpenCvCuda = false;
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            std::cout << "OpenCV CUDA skipped: no CUDA device found by OpenCV.\n";
        } else {
            hasOpenCvCuda = true;
            cv::cuda::setDevice(0);
            cv::cuda::Stream cvStream;
            cv::cuda::GpuMat dSrc, dDst;

            {
                Timer t("OpenCV CUDA H2D upload(gray)");
                dSrc.upload(gray, cvStream);
                cvStream.waitForCompletion();
            }

            for (int w = 0; w < warmup; ++w) {
                cv::cuda::warpAffine(dSrc, dDst, rot2x3, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                     cv::Scalar(0), cvStream);
            }
            cvStream.waitForCompletion();

            {
                Timer t("OpenCV CUDA Benchmark (WarpAffine)");
                cv::cuda::warpAffine(dSrc, dDst, rot2x3, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                     cv::Scalar(0), cvStream);
                cvStream.waitForCompletion();
            }

            {
                Timer t("OpenCV CUDA D2H download(warped)");
                dDst.download(cvCudaWarped, cvStream);
                cvStream.waitForCompletion();
            }

            const std::string outCvCuda = "warp_affine_opencv_cuda_5deg.png";
            const bool okCvCuda = cv::imwrite(outCvCuda, cvCudaWarped);
            std::cout << "Saved opencv cuda warped image: " << outCvCuda << " (" << (okCvCuda ? "ok" : "failed") << ")"
                      << std::endl;
        }
    } catch (const cv::Exception &e) {
        std::cout << "OpenCV CUDA skipped: " << e.what() << "\n";
    }

    PrintSectionHeader(3, "CV-CUDA");
    // Use the same affine matrix as OpenCV to keep transform semantics aligned.
    NVCVAffineTransform xform;
    BuildRotationXform(width, height, angleDeg, xform);

    cudaStream_t stream = nullptr;
    try {
        CUDA_CHECK(cudaStreamCreate(&stream));

        nvcv::Tensor inTensor({{1, height, width, 1}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor outTensor({{1, height, width, 1}, "NHWC"}, nvcv::TYPE_U8);

        auto inDataOpt = inTensor.exportData<nvcv::TensorDataStridedCuda>();
        auto outDataOpt = outTensor.exportData<nvcv::TensorDataStridedCuda>();
        if (!inDataOpt || !outDataOpt) {
            std::cerr << "Failed to export tensor data as CUDA strided buffers." << std::endl;
            if (stream) {
                cudaStreamDestroy(stream);
            }
            return 1;
        }

        const int64_t inRowStride = inDataOpt->stride(1);
        const int64_t outRowStride = outDataOpt->stride(1);

        {
            Timer t("CV-CUDA H2D upload(gray)");
            CUDA_CHECK(cudaMemcpy2DAsync(inDataOpt->basePtr(), inRowStride, gray.data, gray.step,
                                         static_cast<size_t>(width), height, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cvcuda::WarpAffine warpOp(1);

        for (int w = 0; w < warmup; ++w) {
            warpOp(stream, inTensor, outTensor, xform, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,
                   make_float4(0.f, 0.f, 0.f, 0.f));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            Timer t("CV-CUDA Benchmark (WarpAffine)");
            warpOp(stream, inTensor, outTensor, xform, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,
                   make_float4(0.f, 0.f, 0.f, 0.f));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cv::Mat cvcudaWarped(height, width, CV_8UC1);
        {
            Timer t("CV-CUDA D2H download(gray)");
            CUDA_CHECK(cudaMemcpy2DAsync(cvcudaWarped.data, cvcudaWarped.step, outDataOpt->basePtr(), outRowStride,
                                         static_cast<size_t>(width), height, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        const std::string outWarped = "warp_affine_cvcuda_5deg.png";
        const bool okWarp = cv::imwrite(outWarped, cvcudaWarped);
        std::cout << "Saved cvcuda warped image: " << outWarped << " (" << (okWarp ? "ok" : "failed") << ")"
                  << std::endl;

        PrintSectionHeader(4, "Compare");
        if (hasOpenCvCuda) {
            PrintCompareResult("Compare OpenCV CPU vs OpenCV CUDA", CompareImagesU8(cvWarped, cvCudaWarped));
        }
        PrintCompareResult("Compare OpenCV CPU vs CV-CUDA", CompareImagesU8(cvWarped, cvcudaWarped));

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (stream) {
            cudaStreamDestroy(stream);
        }
        return 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
