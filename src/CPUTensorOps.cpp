// Created by mac on 2024/7/31.
//
#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>
#include <numeric>
#include <immintrin.h>

namespace Breeze {
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::add(const Tensor<T>& a, const Tensor<T>& b) const {
        // 验证两个张量的维度是否兼容
        if (a.get_shape() != b.get_shape()) {
            throw std::invalid_argument("Tensor shapes must be equal for addition.");
        }

        // 创建一个新的张量来存储结果，形状与输入张量相同
        auto result = std::make_shared<CPUTensor<T>>(a.get_shape());

        // 使用 std::accumulate 计算总元素数
        const size_t num_elements = std::accumulate(
            a.get_shape().begin(), a.get_shape().end(),
            static_cast<size_t>(1), std::multiplies<size_t>()
        );

        // 使用 BLAS 函数进行加法运算
        T alpha = 1.0;
        if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_saxpy(num_elements, alpha, b.data(), 1, result->data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_daxpy(num_elements, alpha, b.data(), 1, result->data(), 1);
        } else {
            // 对于非浮点类型，使用标准 C++ 循环
            const T* a_data = a.data();
            const T* b_data = b.data();
            T* result_data = result->data();
#pragma omp parallel for
            for (size_t i = 0; i < num_elements; ++i) {
                result_data[i] = a_data[i] + b_data[i];
            }
        }

        return result;
    }


    // 减法实现
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::subtract(const Tensor<T>& a, const Tensor<T>& b) const {
        if (a.get_shape() != b.get_shape()) {
            throw std::invalid_argument("Tensor shapes must be equal for subtraction.");
        }

        auto result = std::make_shared<CPUTensor<T>>(a.get_shape());

        const size_t num_elements = std::accumulate(
            a.get_shape().begin(), a.get_shape().end(),
            static_cast<size_t>(1), std::multiplies<>()
        );

        T alpha = -1.0;
        if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_saxpy(num_elements, alpha, b.data(), 1, result->data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_daxpy(num_elements, alpha, b.data(), 1, result->data(), 1);
        } else {
            const T* a_data = a.data();
            const T* b_data = b.data();
            T* result_data = result->data();
#pragma omp parallel for
            for (size_t i = 0; i < num_elements; ++i) {
                result_data[i] = a_data[i] - b_data[i];
            }
        }

        return result;
    }


    // 乘法实现（元素wise乘法，不是矩阵乘法）
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::multiply(const Tensor<T>& a, const Tensor<T>& b) const {
        if (a.get_shape() != b.get_shape()) {
            throw std::invalid_argument("Tensor shapes must be equal for element-wise multiplication.");
        }

        auto result = std::make_shared<CPUTensor<T>>(a.get_shape());

        const size_t num_elements = std::accumulate(
            a.get_shape().begin(), a.get_shape().end(),
            static_cast<size_t>(1), std::multiplies<>()
        );

        if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_stbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        num_elements, 0, b.data(), 1, result->data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(num_elements, a.data(), 1, result->data(), 1);
            cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        num_elements, 0, b.data(), 1, result->data(), 1);
        } else {
            const T* a_data = a.data();
            const T* b_data = b.data();
            T* result_data = result->data();
#pragma omp parallel for
            for (size_t i = 0; i < num_elements; ++i) {
                result_data[i] = a_data[i] * b_data[i];
            }
        }

        return result;
    }


    // 除法实现（元素wise除法）
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::divide(const Tensor<T>& a, const Tensor<T>& b) const {
        if (a.get_shape() != b.get_shape()) {
            throw std::invalid_argument("Tensor shapes must be equal for element-wise division.");
        }

        auto result = std::make_shared<CPUTensor<T>>(a.get_shape());
        const size_t num_elements = std::accumulate(a.get_shape().begin(), a.get_shape().end(),
                                                    static_cast<size_t>(1), std::multiplies<size_t>());

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();

        if constexpr (std::is_same_v<T, float>) {
#pragma omp parallel for
            size_t i = 0;
            for (; i + 8 <= num_elements; i += 8) {
                const __m256 a_vec = _mm256_loadu_ps(a_data + i);
                const __m256 b_vec = _mm256_loadu_ps(b_data + i);
                if (_mm256_movemask_ps(_mm256_cmp_ps(b_vec, _mm256_setzero_ps(), _CMP_EQ_OQ)) != 0) {
                    throw std::runtime_error("Division by zero encountered.");
                }
                __m256 result_vec = _mm256_div_ps(a_vec, b_vec);
                _mm256_storeu_ps(result_data + i, result_vec);
            }
            // 处理剩余的元素
            for (; i < num_elements; ++i) {
                if (b_data[i] == 0) throw std::runtime_error("Division by zero encountered.");
                result_data[i] = a_data[i] / b_data[i];
            }
        } else if constexpr (std::is_same_v<T, double>) {
#pragma omp parallel for
            size_t i = 0;
            for (; i + 4 <= num_elements; i += 4) {
                const __m256d a_vec = _mm256_loadu_pd(a_data + i); // 如果数据对齐，可以使用 _mm256_load_pd
                const __m256d b_vec = _mm256_loadu_pd(b_data + i);
                if (_mm256_movemask_pd(_mm256_cmp_pd(b_vec, _mm256_setzero_pd(), _CMP_EQ_OQ)) != 0) {
                    throw std::runtime_error("Division by zero encountered.");
                }
                const __m256d result_vec = _mm256_div_pd(a_vec, b_vec);
                _mm256_storeu_pd(result_data + i, result_vec);
            }

            // 处理剩余的元素
            for (; i < num_elements; ++i) {
                if (b_data[i] == 0) throw std::runtime_error("Division by zero encountered.");
                result_data[i] = a_data[i] * (1.0 / b_data[i]);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < num_elements; ++i) {
                if (b_data[i] == 0) throw std::runtime_error("Division by zero encountered.");
                result_data[i] = a_data[i] / b_data[i];
            }
        }

        return result;
    }


    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) const {
        // Get shapes of tensors a and b
        const std::vector<size_t>& a_shape = a.get_shape();
        const std::vector<size_t>& b_shape = b.get_shape();

        // Check for correct dimensions
        if (a_shape.size() < 2 || b_shape.size() < 2) {
            throw std::invalid_argument("Input tensors must have at least two dimensions for matrix multiplication.");
        }

        if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
            throw std::invalid_argument("The inner dimensions must match for matrix multiplication.");
        }

        // Determine the output shape
        std::vector<size_t> result_shape = a_shape;
        result_shape[result_shape.size() - 1] = b_shape[b_shape.size() - 1];

        // Allocate result tensor
        auto result = std::make_shared<CPUTensor<T>>(result_shape);

        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Compute the strides for each tensor
        const std::vector<size_t> a_strides = compute_strides(a_shape);
        const std::vector<size_t> b_strides = compute_strides(b_shape);
        const std::vector<size_t> result_strides = compute_strides(result_shape);

        // Call the recursive matrix multiplication
        multiply_non_recursive(a.data(), b.data(), result->data(), a_shape, b_shape, a_strides, b_strides, result_strides);

        return result;
    }


    template<typename T>
    void CPUTensorOps<T>::multiply_non_recursive(const T* a, const T* b, T* result, const std::vector<size_t>& a_shape,
                                                 const std::vector<size_t>& b_shape,
                                                 const std::vector<size_t>& a_strides, const std::vector<size_t>& b_strides,
                                                 const std::vector<size_t>& result_strides) const {
        const size_t depth = a_shape.size() - 2;
        size_t m = a_shape[depth];
        size_t k = a_shape[depth + 1];
        size_t n = b_shape[depth + 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Calculate the number of 2D matrices (product of dimensions up to 'depth')
        size_t num_matrices = 1;
        for (size_t i = 0; i < depth; ++i) {
            num_matrices *= a_shape[i];
        }

        // Parallelize using OpenMP
#pragma omp parallel for
        for (size_t idx = 0; idx < num_matrices; ++idx) {
            size_t a_offset = 0;
            size_t b_offset = 0;
            size_t result_offset = 0;
            size_t temp_idx = idx;

            // Calculate the offsets for each matrix slice
            for (size_t i = 0; i < depth; ++i) {
                const size_t index = temp_idx % a_shape[i];
                a_offset += index * a_strides[i];
                b_offset += index * b_strides[i];
                result_offset += index * result_strides[i];
                temp_idx /= a_shape[i];
            }

            const T* a_sub = a + a_offset;
            const T* b_sub = b + b_offset;
            T* result_sub = result + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            }
        }
}

    template<typename T>
    [[nodiscard]] std::vector<size_t> CPUTensorOps<T>::compute_strides(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape vector cannot be empty.");
        }
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        std::vector<size_t> strides(shape.size(), 1);
        const auto s_size = static_cast<int>(shape.size());
        for (int i = s_size - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
}

    template<typename T>
    void CPUTensorOps<T>::broadcastTensors(Tensor<T>& a, Tensor<T>& b) {
        const std::vector<size_t>& shape1 = a.get_shape();
        const std::vector<size_t>& shape2 = b.get_shape();

        // 计算目标形状
        std::vector<size_t> targetShape;
        const int maxDims = static_cast<int>(std::max(shape1.size(), shape2.size()));
        targetShape.resize(maxDims);

        for (int i = 0; i < maxDims; ++i) {
            //如果在当前没有维度 就认为是 1
            const size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
            const size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::runtime_error("Incompatible shapes for broadcasting");
            }
            targetShape[maxDims - 1 - i] = std::max(static_cast<int>(dim1), static_cast<int>(dim2));
        }

        // 计算目标大小和步长
        const size_t targetSize = std::accumulate(targetShape.begin(), targetShape.end(), 1, std::multiplies<>());
        std::vector<int> targetStrides(maxDims);
        int stride = 1;
        for (int i = maxDims - 1; i >= 0; --i) {
            targetStrides[i] = stride;
            stride *= static_cast<int>(targetShape[i]);
        }

        // 计算输入向量的步长
        std::vector<int> strides1(maxDims, 0), strides2(maxDims, 0);
        stride = 1;
        for (int i = static_cast<int>(shape1.size()) - 1; i >= 0; --i) {
            strides1[maxDims - shape1.size() + i] = shape1[i] == 1 ? 0 : stride;
            stride *= static_cast<int>(shape1[i]);
        }
        stride = 1;
        for (int i = static_cast<int>(shape2.size()) - 1; i >= 0; --i) {
            strides2[maxDims - shape2.size() + i] = (shape2[i] == 1) ? 0 : stride;
            stride *=  static_cast<int>(shape2[i]);
        }

        // 执行广播
        auto result1 = std::make_shared<Blob<T>>(targetShape);
        auto result2 = std::make_shared<Blob<T>>(targetShape);
        std::vector indices(maxDims, 0);

        const auto data_a = static_cast<const T*>(a.data());
        const auto data_b = static_cast<const T*>(b.data());
        for (int i = 0; i < targetSize; ++i) {
            int index1 = 0, index2 = 0;
            //计算 每个被复制变量位置 如果1为 只能在 +0 位置 如果大于1 就是 0...j 每次取完递增++indices[j]
            for (int j = 0; j < maxDims; ++j) {
                index1 += indices[j] * strides1[j];
                index2 += indices[j] * strides2[j];
            }
            result1->getData()[i] = data_a[index1];
            result2->getData()[i] = data_b[index2];

            // 更新索引
            for (int j = maxDims - 1; j >= 0; --j) {
                //当前循环维度 小于目标维度
                if (++indices[j] < targetShape[j]) {
                    break;
                }
                //从第0个开始
                indices[j] = 0;
            }
        }
        a.setData(result1);
        b.setData(result2);
}

    // Explicit template instantiation
    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
};


