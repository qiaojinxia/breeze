cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect operating system and architecture
if(APPLE)
    set(IS_MACOS TRUE)
elseif(WIN32)
    set(IS_WINDOWS TRUE)
else()
    set(IS_LINUX TRUE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(IS_ARM TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(IS_X86_64 TRUE)
endif()

if(IS_MACOS)
    if(IS_ARM)
        set(HOMEBREW_PREFIX "/opt/homebrew")
    else()
        set(HOMEBREW_PREFIX "/usr/local")
    endif()
    message(STATUS "Using Homebrew prefix: ${HOMEBREW_PREFIX}")
else()
    message(WARNING "Unsupported platform")
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3)
    if(IS_X86_64)
        add_compile_options(-mavx2)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /O2)
    if(IS_X86_64)
        add_compile_options(/arch:AVX2)
    endif()
endif()

# Set SIMD instruction set and related source files
if(IS_ARM)
    add_definitions(-DUSE_NEON)
    set(SIMD_HEADERS src/platform/VectorizedNeon.h)
elseif(IS_X86_64)
    add_definitions(-DUSE_AVX2)
    set(SIMD_HEADERS src/platform/VectorizedAvx2.h)
else()
    message(WARNING "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Using scalar operations.")
endif()

# Find required packages

if(IS_MACOS)
    # Use Clang as the compiler
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    # Manually specify SLEEF installation directory if not found
    set(SLEEF_ROOT "${HOMEBREW_PREFIX}/opt/sleef")
    find_package(SLEEF REQUIRED PATHS ${SLEEF_ROOT})
    set(SLEEF_LIB "${HOMEBREW_PREFIX}/lib/libsleef.3.dylib")

    set(OpenBLAS_ROOT "${HOMEBREW_PREFIX}/opt/openblas")
    find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${OpenBLAS_ROOT}/include NO_DEFAULT_PATH)
    find_library(OpenBLAS_LIB NAMES openblas PATHS ${OpenBLAS_ROOT}/lib NO_DEFAULT_PATH)

    if(NOT OpenBLAS_INCLUDE_DIR OR NOT OpenBLAS_LIB)
        message(FATAL_ERROR "OpenBLAS not found. Please install it using Homebrew: brew install openblas")
    endif()

    message(STATUS "OpenBLAS include dir: ${OpenBLAS_INCLUDE_DIR}")
    message(STATUS "OpenBLAS library: ${OpenBLAS_LIB}")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${HOMEBREW_PREFIX}/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_INCLUDE_DIRS "${HOMEBREW_PREFIX}/opt/libomp/include")


endif()

find_package(Armadillo REQUIRED)
find_package(SLEEF REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(OpenBLAS REQUIRED)
# Define source files
set(SOURCES
        src/main.cpp
        src/node.cpp
        src/CPUTensorOps.cpp
        src/CPUTensor.cpp
        src/TensorIterator.cpp
)

# Define header files
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
        src/platform/SIMDFactory.h
        ${SIMD_HEADERS}
        src/ScalarType.h
)

# Add executable
add_executable(breeze ${SOURCES} ${HEADERS})

# Add compile definitions
target_compile_definitions(breeze PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
)

# Set include directories
target_include_directories(breeze PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIR}
        ${SLEEF_INCLUDE_DIRS}
        ${OpenMP_CXX_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(breeze PRIVATE
        ${ARMADILLO_LIBRARIES}
        ${OpenBLAS_LIB}
        OpenMP::OpenMP_CXX
        ${SLEEF_LIB}
)

# Ensure C++17 is used
target_compile_features(breeze PRIVATE cxx_std_17)