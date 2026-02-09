cmake_minimum_required(VERSION 3.18)
project(cvcuda_benchmarks)

set(CMAKE_CXX_STANDARD 17)

# --- 1. 查找依赖 ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# --- 2. 设置 CV-CUDA 路径 ---
set(CVCUDA_ROOT "/opt/nvidia/cvcuda0")
if(DEFINED ENV{CVCUDA_ROOT})
    set(CVCUDA_ROOT $ENV{CVCUDA_ROOT})
endif()

# --- 3. 配置编译目标 (在这里切换文件名) ---
# 你只需要修改下面这一行，就能编译不同的程序
set(TARGET_SRC "channel.cpp") 
set(TARGET_NAME "my_app")

# 定义可执行文件
add_executable(${TARGET_NAME} ${TARGET_SRC})

# --- 4. 配置包含路径 (Target-based) ---
target_include_directories(${TARGET_NAME} PRIVATE 
    ${CVCUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

# --- 5. 配置库和链接 ---
target_link_directories(${TARGET_NAME} PRIVATE ${CVCUDA_ROOT}/lib)

# 设置 RPATH 确保运行时能找到 .so
set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH "${CVCUDA_ROOT}/lib")

# 链接所有依赖
target_link_libraries(${TARGET_NAME} PRIVATE 
    cvcuda 
    nvcv_types 
    CUDA::cudart
    ${OpenCV_LIBS}
)

# 打印信息确认
message(STATUS "Now building: ${TARGET_SRC} as ${TARGET_NAME}")