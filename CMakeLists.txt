cmake_minimum_required(VERSION 3.18)
project(cvcuda_benchmarks)

# ==============================================================================
# 1. 编译目标设置 (在这里切换文件名)
# ==============================================================================
set(TARGET_SRC "hello_world.cpp") 
set(TARGET_NAME "my_app")

set(CMAKE_CXX_STANDARD 17)

# ==============================================================================
# 2. 【核心依赖】 CUDA Toolkit (必选)
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# 3. 【模块 A】 OpenCV (可选)
#    如果不需要 OpenCV，请注释掉下面这两行以及下方的链接部分
# ==============================================================================
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
add_executable(${TARGET_NAME} ${TARGET_SRC})

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
)

# 只要编译 examples/ 下的源码，就加 examples 头文件路径（如 common.hpp）
if(TARGET_SRC MATCHES "^examples/.*")
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
    )
endif()

# ==============================================================================
# 7. 配置库链接 (Linking Libraries)
# ==============================================================================

# [CV-CUDA 专属] 设置库搜索目录和运行时 RPATH
target_link_directories(${TARGET_NAME} PRIVATE ${CVCUDA_ROOT}/lib)
set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH "${CVCUDA_ROOT}/lib")

target_link_libraries(${TARGET_NAME} PRIVATE 
    # --- [CUDA] 核心运行时 ---
    CUDA::cudart

    # --- [TRT] TensorRT 推理引擎库 ---
    nvinfer

    # --- [OpenCV] 图像处理库 ---
    ${OpenCV_LIBS}

    # --- [CV-CUDA] GPU 加速预处理库 ---
    cvcuda 
    nvcv_types 
)

# ==============================================================================
# 8. 状态打印确认
# ==============================================================================
message(STATUS "Build Configuration Summary:")
message(STATUS "  Target        : ${TARGET_NAME}")
message(STATUS "  Source        : ${TARGET_SRC}")
message(STATUS "  CUDA version  : ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV found  : ${OpenCV_VERSION}")
message(STATUS "  CVCUDA path   : ${CVCUDA_ROOT}")
