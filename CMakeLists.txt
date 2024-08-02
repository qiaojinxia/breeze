cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)


# 查找 Armadillo 库
find_package(Armadillo REQUIRED)

find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/local/opt/openblas/include)
find_library(OpenBLAS_LIB NAMES openblas PATHS /usr/local/opt/openblas/lib)


add_executable(breeze src/main.cpp
        src/node.cpp
        src/node.h
        src/loss_functions.h
        src/AlignedDataBlob.cpp
        src/AlignedDataBlob.h
        src/Tensor.h
        src/TensorOps.h
        src/CPUTensorOps.cpp
        src/CPUTensorOps.h
        src/CPUTensor.cpp
        src/CPUTensor.h)

if(APPLE)
    add_definitions(-DACCELERATE_NEW_LAPACK)
    add_definitions(-DACCELERATE_LAPACK_ILP64)
endif()

# 链接 Armadillo 库
target_link_libraries(breeze ${ARMADILLO_LIBRARIES})
target_link_libraries(breeze ${OpenBLAS_LIB})

target_include_directories(breeze PRIVATE ${OpenBLAS_INCLUDE_DIR})
# 包含 Armadillo 头文件
target_include_directories(breeze PRIVATE ${ARMADILLO_INCLUDE_DIRS})