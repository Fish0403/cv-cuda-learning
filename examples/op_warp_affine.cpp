// op_warp_affine.cpp
// OpenCV vs CV-CUDA affine rotation demo:
// load grayscale image, rotate by fixed 5 degrees, save CV-CUDA output image.

#include <cvcuda/OpWarpAffine.hpp>

#include "common.hpp"

static void BuildInverseRotationXform(int width, int height, double angleDeg, NVCVAffineTransform xform)
{
    const cv::Point2f center(width * 0.5f, height * 0.5f);
    const cv::Mat rot2x3 = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    cv::Mat invRot2x3;
    cv::invertAffineTransform(rot2x3, invRot2x3);

    xform[0] = static_cast<float>(invRot2x3.at<double>(0, 0));
    xform[1] = static_cast<float>(invRot2x3.at<double>(0, 1));
    xform[2] = static_cast<float>(invRot2x3.at<double>(0, 2));
    xform[3] = static_cast<float>(invRot2x3.at<double>(1, 0));
    xform[4] = static_cast<float>(invRot2x3.at<double>(1, 1));
    xform[5] = static_cast<float>(invRot2x3.at<double>(1, 2));
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
    std::cout << "Input(gray): " << width << "x" << height << std::endl;

    const double angleDeg = 5.0;
    const cv::Point2f center(width * 0.5f, height * 0.5f);
    const cv::Mat rot2x3 = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    // ========================
    // OpenCV section
    // ========================
    cv::Mat cvWarped(height, width, CV_8UC1);
    {
        Timer t("OpenCV warpAffine (5 deg)");
        cv::warpAffine(gray, cvWarped, rot2x3, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    }

    // ========================
    // CV-CUDA section
    // ========================
    // CV-CUDA warp uses destination-to-source mapping.
    // Build inverse matrix from the same OpenCV rotation settings.
    NVCVAffineTransform xform;
    BuildInverseRotationXform(width, height, angleDeg, xform);

    cudaStream_t stream = nullptr;
    try {
        CUDA_CHECK(cudaStreamCreate(&stream));

        nvcv::Tensor inTensor({{height, width, 1}, "HWC"}, nvcv::TYPE_U8);
        nvcv::Tensor outTensor({{height, width, 1}, "HWC"}, nvcv::TYPE_U8);

        auto inDataOpt = inTensor.exportData<nvcv::TensorDataStridedCuda>();
        auto outDataOpt = outTensor.exportData<nvcv::TensorDataStridedCuda>();
        if (!inDataOpt || !outDataOpt) {
            std::cerr << "Failed to export tensor data as CUDA strided buffers." << std::endl;
            if (stream) {
                cudaStreamDestroy(stream);
            }
            return 1;
        }

        const int64_t inRowStride = inDataOpt->stride(0);
        const int64_t outRowStride = outDataOpt->stride(0);

        {
            Timer t("H2D upload(gray)");
            CUDA_CHECK(cudaMemcpy2D(inDataOpt->basePtr(), inRowStride, gray.data, gray.step,
                                    static_cast<size_t>(width), height, cudaMemcpyHostToDevice));
        }

        cvcuda::WarpAffine warpOp(1);

        // Warmup to exclude one-time setup overhead from measured run.
        warpOp(stream, inTensor, outTensor, xform, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,
               make_float4(0.f, 0.f, 0.f, 0.f));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            Timer t("CVCUDA WarpAffine (5 deg)");
            warpOp(stream, inTensor, outTensor, xform, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,
                   make_float4(0.f, 0.f, 0.f, 0.f));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cv::Mat cvcudaWarped(height, width, CV_8UC1);
        {
            Timer t("D2H download(cvcuda)");
            CUDA_CHECK(cudaMemcpy2D(cvcudaWarped.data, cvcudaWarped.step, outDataOpt->basePtr(), outRowStride,
                                    static_cast<size_t>(width), height, cudaMemcpyDeviceToHost));
        }

        const std::string outWarped = "warp_affine_cvcuda_5deg.png";
        const bool okWarp = cv::imwrite(outWarped, cvcudaWarped);
        std::cout << "Saved cvcuda warped image: " << outWarped << " (" << (okWarp ? "ok" : "failed") << ")"
                  << std::endl;

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
