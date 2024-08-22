cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Armadillo REQUIRED)

# 查找 OpenBLAS
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
        # Apple Silicon (M1, M2, etc.)
        set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        # Intel-based Mac
        set(OpenBLAS_ROOT "/usr/local/opt/openblas")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${OpenBLAS_ROOT}/include NO_DEFAULT_PATH)
    find_library(OpenBLAS_LIB NAMES openblas PATHS ${OpenBLAS_ROOT}/lib NO_DEFAULT_PATH)
else()
    find_package(OpenBLAS REQUIRED)
endif()

# 查找 OpenMP 库
if(OpenMP_CXX_FOUND)
    # OpenMP 找到了，使用 CMake 提供的变量
    set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY OpenMP::OpenMP_CXX)
else()
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            # Apple Silicon (M1, M2, etc.)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            # Intel-based Mac
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/18.1.8/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/18.1.8/lib/libomp.dylib")
        endif()
    else()
        # For other platforms, you may need to adjust these settings
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "gomp")
        find_library(OpenMP_gomp_LIBRARY NAMES gomp)
        set(OpenMP_omp_LIBRARY ${OpenMP_gomp_LIBRARY})
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# 定义源文件
set(SOURCES
        src/main.cpp
        src/node.cpp
        src/CPUTensorOps.cpp
        src/CPUTensor.cpp
        src/common/Utils.h
        src/tests/test_cases.h
)

# 定义头文件（可选）
set(HEADERS
        src/node.h
        src/loss_functions.h
        src/Tensor.h
        src/TensorOps.h
        src/CPUTensorOps.h
        src/CPUTensor.h
        src/TensorStorage.h
        src/common/Macro.h
)

# 添加可执行文件
add_executable(breeze ${SOURCES})


# 添加编译定义
target_compile_definitions(breeze PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64

)

target_compile_options(breeze PRIVATE ${OpenMP_FLAGS} "-Dmavx" )


# 设置包含目录
target_include_directories(breeze PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIR}
        ${OpenMP_CXX_INCLUDE_DIRS} # 添加 OpenMP 的包含目录
)

# 链接库
target_link_libraries(breeze PRIVATE
        ${ARMADILLO_LIBRARIES}
        ${OpenBLAS_LIB}
        OpenMP::OpenMP_CXX # 添加 OpenMP 链接
)

# 如果使用 C++17 的特性，确保编译器支持
target_compile_features(breeze PRIVATE cxx_std_17)
