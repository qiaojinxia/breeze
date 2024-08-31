cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Armadillo REQUIRED)

# 设置 SIMD 指令集和相关源文件
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_definitions(-DUSE_NEON)
    set(SIMD_SOURCES src/platform/NEONOps.cpp)
    set(SIMD_HEADERS src/platform/NEONOps.h)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    add_definitions(-DUSE_AVX2)
    set(SIMD_SOURCES src/platform/AVX2Ops.cpp)
    set(SIMD_HEADERS src/platform/AVX2Ops.h)
else()
    message(WARNING "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Using scalar operations.")
endif()

# 查找 OpenBLAS
if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(OpenBLAS_ROOT "/usr/local/opt/openblas")
    endif()
    find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${OpenBLAS_ROOT}/include NO_DEFAULT_PATH)
    find_library(OpenBLAS_LIB NAMES openblas PATHS ${OpenBLAS_ROOT}/lib NO_DEFAULT_PATH)
else()
    find_package(OpenBLAS REQUIRED)
endif()

# 查找 OpenMP 库
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY OpenMP::OpenMP_CXX)
else()
    if(APPLE)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY omp)
        endif()

        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY omp)
        endif()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/opt/libomp/include")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
            set(OpenMP_CXX_INCLUDE_DIRS "/usr/local/opt/libomp/include")
        endif()
    else()
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "gomp")
        find_library(OpenMP_LIBS NAMES gomp)
        set(OpenMP_LIBS ${OpenMP_gomp_LIBRARY})
    endif()
endif()

find_package(OpenMP REQUIRED)
# 定义源文件
set(SOURCES
        src/main.cpp
        src/node.cpp
        src/CPUTensorOps.cpp
        src/CPUTensor.cpp
        src/TensorIterator.cpp
        ${SIMD_SOURCES}
)

# 定义头文件
set(HEADERS
        src/node.h
        src/loss_functions.h
        src/Tensor.h
        src/TensorOps.h
        src/CPUTensorOps.h
        src/CPUTensor.h
        src/TensorStorage.h
        src/common/Macro.h
        src/common/Utils.h
        src/tests/test_cases.h
        src/TensorIterator.h
        src/platform/SIMDOps.h
        src/platform/SIMDFactory.h
        ${SIMD_HEADERS}
        src/platform/Vectorized.cpp
        src/platform/Vectorized.h
)

# 添加可执行文件
add_executable(breeze ${SOURCES} ${HEADERS})

# 添加编译定义
target_compile_definitions(breeze PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
)

# 设置编译选项


if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_options(breeze PRIVATE "-mavx2")
endif()


# 设置包含目录
target_include_directories(breeze PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIR}
        ${OpenMP_CXX_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接库
target_link_libraries(breeze PRIVATE
        ${ARMADILLO_LIBRARIES}
        ${OpenBLAS_LIB}
        OpenMP::OpenMP_CXX

)

# 确保使用 C++17
target_compile_features(breeze PRIVATE cxx_std_17)