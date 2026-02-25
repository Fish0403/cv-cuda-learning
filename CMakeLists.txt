cmake_minimum_required(VERSION 3.18)
project(cvcuda_benchmarks)

# ==============================================================================
# 1. 编译目标设置
# ==============================================================================
set(EXAMPLE_SOURCES
    hello_world.cpp
    # examples/op_resize.cpp
    # examples/op_average_blur.cpp
    # examples/op_warp_affine.cpp
    # trt_preprocessing_benchmark.cpp
    # trt_cvcuda_pipeline_overlap_benchmark.cpp
)

set(CMAKE_CXX_STANDARD 17)

# ==============================================================================
# 2. 【核心依赖】 CUDA Toolkit (必选)
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# 3. 【模块 A】 OpenCV (可选)
#    如果不需要 OpenCV，请注释掉下面这两行以及下方的链接部分
# ==============================================================================
# OpenCV CPU 版（系统包）
# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "CPU OpenCV")

# OpenCV CUDA 版（源码安装到 /usr/local）
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4" CACHE PATH "CUDA OpenCV")
find_package(OpenCV REQUIRED)

# ==============================================================================
# 4. 【模块 B】 CV-CUDA (可选)
#    设置路径并确认包含路径
# ==============================================================================
set(CVCUDA_ROOT "/opt/nvidia/cvcuda0") 
if(DEFINED ENV{CVCUDA_ROOT})
    set(CVCUDA_ROOT $ENV{CVCUDA_ROOT})
endif()

# ==============================================================================
# 5. 配置执行目标
# ==============================================================================
function(add_example_target SOURCE_FILE)
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    # ==============================================================================
    # 6. 配置包含路径 (Header Search Paths)
    # ==============================================================================
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        # --- [TRT] TensorRT 头文件 ---
        /usr/include/x86_64-linux-gnu
        # --- [OpenCV] 头文件 ---
        ${OpenCV_INCLUDE_DIRS}
        # --- [CV-CUDA] 头文件 ---
        ${CVCUDA_ROOT}/include
        # examples 下公共头文件（如 common.hpp）
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
    )

    # ==============================================================================
    # 7. 配置库链接 (Linking Libraries)
    # ==============================================================================
    target_link_directories(${TARGET_NAME} PRIVATE ${CVCUDA_ROOT}/lib)
    set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH "${CVCUDA_ROOT}/lib")

    target_link_libraries(${TARGET_NAME} PRIVATE
        CUDA::cudart
        nvinfer
        ${OpenCV_LIBS}
        cvcuda
        nvcv_types
    )
endfunction()

foreach(EXAMPLE_SRC IN LISTS EXAMPLE_SOURCES)
    add_example_target(${EXAMPLE_SRC})
endforeach()

# ==============================================================================
# 8. 状态打印确认
# ==============================================================================
message(STATUS "Build Configuration Summary:")
message(STATUS "  Targets       : ${EXAMPLE_SOURCES}")
message(STATUS "  CUDA version  : ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV found  : ${OpenCV_VERSION}")
message(STATUS "  OpenCV_DIR    : ${OpenCV_DIR}")
message(STATUS "  CVCUDA path   : ${CVCUDA_ROOT}")
