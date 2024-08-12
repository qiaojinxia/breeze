// Created by mac on 2024/7/31.
//
#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>
#include <immintrin.h>

namespace Breeze {
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::add(const Tensor<T>& a, const Tensor<T>& b) const {
        auto [a_strides, b_strides, target_shape] = this->broadcastTensors(a, b);

        auto result = std::make_shared<CPUTensor<T>>(Shape(target_shape));

        size_t outer_dim = 1;
        for (size_t i = 0; i < target_shape.size() - 1; ++i) {
            outer_dim *= target_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        const size_t inner_dim = target_shape.back();

#pragma omp parallel for
        for (size_t i = 0; i < outer_dim; ++i) {
            std::vector<size_t> coords(target_shape.size() - 1);
            size_t temp = i;
            for (int k = static_cast<int>(target_shape.size()) - 2; k >= 0; --k) {
                coords[k] = temp % target_shape[k];
                temp /= target_shape[k];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t k = 0; k < coords.size(); ++k) {
                a_offset += coords[k] * a_strides[k] * a_steps[k];
                b_offset += coords[k] * b_strides[k] * b_steps[k];
                result_offset += coords[k] * result->get_strides()[k];
            }

            int a_inc = a_strides.back() * a_steps.back();
            int b_inc = b_strides.back() * b_steps.back();

            if constexpr (std::is_same_v<T, float>) {
                cblas_scopy(inner_dim, a_data + a_offset, a_inc, result_data + result_offset, 1);
                cblas_saxpy(inner_dim, 1.0f, b_data + b_offset, b_inc, result_data + result_offset, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dcopy(inner_dim, a_data + a_offset, a_inc, result_data + result_offset, 1);
                cblas_daxpy(inner_dim, 1.0, b_data + b_offset, b_inc, result_data + result_offset, 1);
            } else {
                for (size_t j = 0; j < inner_dim; ++j) {
                    result_data[result_offset + j * 1] =
                        a_data[a_offset + j * a_inc] + b_data[b_offset + j * b_inc];
                }
            }
        }

        return result;
    }


    // 减法实现
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::subtract(const Tensor<T>& a, const Tensor<T>& b) const {
        auto [a_strides, b_strides, target_shape] = this->broadcastTensors(a, b);
        auto result = std::make_shared<CPUTensor<T>>(Shape(target_shape));

        size_t outer_dim = 1;
        for (size_t i = 0; i < target_shape.size() - 1; ++i) {
            outer_dim *= target_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        const size_t inner_dim = target_shape.back();

#pragma omp parallel for
        for (size_t i = 0; i < outer_dim; ++i) {
            std::vector<size_t> coords(target_shape.size() - 1);
            size_t temp = i;
            for (int k = static_cast<int>(target_shape.size()) - 2; k >= 0; --k) {
                coords[k] = temp % target_shape[k];
                temp /= target_shape[k];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t k = 0; k < coords.size(); ++k) {
                a_offset += coords[k] * a_strides[k] * a_steps[k];
                b_offset += coords[k] * b_strides[k] * b_steps[k];
                result_offset += coords[k] * result->get_strides()[k];
            }

            int a_inc = a_strides.back() * a_steps.back();
            int b_inc = b_strides.back() * b_steps.back();

            if constexpr (std::is_same_v<T, float>) {
                cblas_scopy(inner_dim, a_data + a_offset, a_inc, result_data + result_offset, 1);
                cblas_saxpy(inner_dim, -1.0f, b_data + b_offset, b_inc, result_data + result_offset, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dcopy(inner_dim, a_data + a_offset, a_inc, result_data + result_offset, 1);
                cblas_daxpy(inner_dim, -1.0, b_data + b_offset, b_inc, result_data + result_offset, 1);
            } else {
                for (size_t j = 0; j < inner_dim; ++j) {
                    result_data[result_offset + j] =
                        a_data[a_offset + j * a_inc] - b_data[b_offset + j * b_inc];
                }
            }
        }

        return result;
}

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::multiply(const Tensor<T>& a, const Tensor<T>& b) const {
        auto [a_strides, b_strides, target_shape] = this->broadcastTensors(a, b);

        auto result = std::make_shared<CPUTensor<T>>(Shape(target_shape));

        size_t outer_dim = 1;
        for (size_t i = 0; i < target_shape.size() - 1; ++i) {
            outer_dim *= target_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        const size_t inner_dim = target_shape.back();

#pragma omp parallel for
        for (size_t i = 0; i < outer_dim; ++i) {
            std::vector<size_t> coords(target_shape.size() - 1);
            size_t temp = i;
            for (int k = static_cast<int>(target_shape.size()) - 2; k >= 0; --k) {
                coords[k] = temp % target_shape[k];
                temp /= target_shape[k];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t k = 0; k < coords.size(); ++k) {
                a_offset += coords[k] * a_strides[k] * a_steps[k];
                b_offset += coords[k] * b_strides[k] * b_steps[k];
                result_offset += coords[k] * result->get_strides()[k];
            }

            const int a_inc = a_strides.back() * a_steps.back();
            const int b_inc = b_strides.back() * b_steps.back();

            if constexpr (std::is_same_v<T, float>) {
                for (size_t j = 0; j < inner_dim; j += 8) {
                    const __m256 a_vec = _mm256_setr_ps(
                        j + 0 < inner_dim ? a_data[a_offset + (j + 0) * a_inc] : 1,
                        j + 1 < inner_dim ? a_data[a_offset + (j + 1) * a_inc] : 1,
                        j + 2 < inner_dim ? a_data[a_offset + (j + 2) * a_inc] : 1,
                        j + 3 < inner_dim ? a_data[a_offset + (j + 3) * a_inc] : 1,
                        j + 4 < inner_dim ? a_data[a_offset + (j + 4) * a_inc] : 1,
                        j + 5 < inner_dim ? a_data[a_offset + (j + 5) * a_inc] : 1,
                        j + 6 < inner_dim ? a_data[a_offset + (j + 6) * a_inc] : 1,
                        j + 7 < inner_dim ? a_data[a_offset + (j + 7) * a_inc] : 1
                    );
                    const __m256 b_vec = _mm256_setr_ps(
                        j + 0 < inner_dim ? b_data[b_offset + (j + 0) * b_inc] : 1,
                        j + 1 < inner_dim ? b_data[b_offset + (j + 1) * b_inc] : 1,
                        j + 2 < inner_dim ? b_data[b_offset + (j + 2) * b_inc] : 1,
                        j + 3 < inner_dim ? b_data[b_offset + (j + 3) * b_inc] : 1,
                        j + 4 < inner_dim ? b_data[b_offset + (j + 4) * b_inc] : 1,
                        j + 5 < inner_dim ? b_data[b_offset + (j + 5) * b_inc] : 1,
                        j + 6 < inner_dim ? b_data[b_offset + (j + 6) * b_inc] : 1,
                        j + 7 < inner_dim ? b_data[b_offset + (j + 7) * b_inc] : 1
                    );
                    __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_data + result_offset + j, result_vec);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                for (size_t j = 0; j < inner_dim; j += 4) {
                    const __m256d a_vec = _mm256_setr_pd(
                        j + 0 < inner_dim ? a_data[a_offset + (j + 0) * a_inc] : 1,
                        j + 1 < inner_dim ? a_data[a_offset + (j + 1) * a_inc] : 1,
                        j + 2 < inner_dim ? a_data[a_offset + (j + 2) * a_inc] : 1,
                        j + 3 < inner_dim ? a_data[a_offset + (j + 3) * a_inc] : 1
                    );
                    const __m256d b_vec = _mm256_setr_pd(
                        j + 0 < inner_dim ? b_data[b_offset + (j + 0) * b_inc] : 1,
                        j + 1 < inner_dim ? b_data[b_offset + (j + 1) * b_inc] : 1,
                        j + 2 < inner_dim ? b_data[b_offset + (j + 2) * b_inc] : 1,
                        j + 3 < inner_dim ? b_data[b_offset + (j + 3) * b_inc] : 1
                    );
                    __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result_data + result_offset + j, result_vec);
                }
            } else {
                for (size_t j = 0; j < inner_dim; ++j) {
                    result_data[result_offset + j] =
                        a_data[a_offset + j * a_inc] * b_data[b_offset + j * b_inc];
                }
            }
        }

        return result;
    }


    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::divide(const Tensor<T>& a, const Tensor<T>& b) const {
        auto [a_strides, b_strides, target_shape] = this->broadcastTensors(a, b);

        auto result = std::make_shared<CPUTensor<T>>(Shape(target_shape));

        size_t outer_dim = 1;
        for (size_t i = 0; i < target_shape.size() - 1; ++i) {
            outer_dim *= target_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        const size_t inner_dim = target_shape.back();

#pragma omp parallel for
        for (size_t i = 0; i < outer_dim; ++i) {
            std::vector<size_t> coords(target_shape.size() - 1);
            size_t temp = i;
            for (int k = static_cast<int>(target_shape.size()) - 2; k >= 0; --k) {
                coords[k] = temp % target_shape[k];
                temp /= target_shape[k];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t k = 0; k < coords.size(); ++k) {
                a_offset += coords[k] * a_strides[k] * a_steps[k];
                b_offset += coords[k] * b_strides[k] * b_steps[k];
                result_offset += coords[k] * result->get_strides()[k];
            }

            const int a_inc = a_strides.back() * a_steps.back();
            const int b_inc = b_strides.back() * b_steps.back();

            if constexpr (std::is_same_v<T, float>) {
                for (size_t j = 0; j < inner_dim; j += 8) {
                    const __m256 a_vec = _mm256_setr_ps(
                        j + 0 < inner_dim ? a_data[a_offset + (j + 0) * a_inc] : 0,
                        j + 1 < inner_dim ? a_data[a_offset + (j + 1) * a_inc] : 0,
                        j + 2 < inner_dim ? a_data[a_offset + (j + 2) * a_inc] : 0,
                        j + 3 < inner_dim ? a_data[a_offset + (j + 3) * a_inc] : 0,
                        j + 4 < inner_dim ? a_data[a_offset + (j + 4) * a_inc] : 0,
                        j + 5 < inner_dim ? a_data[a_offset + (j + 5) * a_inc] : 0,
                        j + 6 < inner_dim ? a_data[a_offset + (j + 6) * a_inc] : 0,
                        j + 7 < inner_dim ? a_data[a_offset + (j + 7) * a_inc] : 0
                    );
                    const __m256 b_vec = _mm256_setr_ps(
                        j + 0 < inner_dim ? b_data[b_offset + (j + 0) * b_inc] : 1,
                        j + 1 < inner_dim ? b_data[b_offset + (j + 1) * b_inc] : 1,
                        j + 2 < inner_dim ? b_data[b_offset + (j + 2) * b_inc] : 1,
                        j + 3 < inner_dim ? b_data[b_offset + (j + 3) * b_inc] : 1,
                        j + 4 < inner_dim ? b_data[b_offset + (j + 4) * b_inc] : 1,
                        j + 5 < inner_dim ? b_data[b_offset + (j + 5) * b_inc] : 1,
                        j + 6 < inner_dim ? b_data[b_offset + (j + 6) * b_inc] : 1,
                        j + 7 < inner_dim ? b_data[b_offset + (j + 7) * b_inc] : 1
                    );
                    if (_mm256_movemask_ps(_mm256_cmp_ps(b_vec, _mm256_setzero_ps(), _CMP_EQ_OQ)) != 0) {
                        throw std::runtime_error("Division by zero encountered.");
                    }
                    __m256 result_vec = _mm256_div_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_data + result_offset + j, result_vec);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                for (size_t j = 0; j < inner_dim; j += 4) {
                    const __m256d a_vec = _mm256_setr_pd(
                        j + 0 < inner_dim ? a_data[a_offset + (j + 0) * a_inc] : 0,
                        j + 1 < inner_dim ? a_data[a_offset + (j + 1) * a_inc] : 0,
                        j + 2 < inner_dim ? a_data[a_offset + (j + 2) * a_inc] : 0,
                        j + 3 < inner_dim ? a_data[a_offset + (j + 3) * a_inc] : 0
                    );
                    const __m256d b_vec = _mm256_setr_pd(
                        j + 0 < inner_dim ? b_data[b_offset + (j + 0) * b_inc] : 1,
                        j + 1 < inner_dim ? b_data[b_offset + (j + 1) * b_inc] : 1,
                        j + 2 < inner_dim ? b_data[b_offset + (j + 2) * b_inc] : 1,
                        j + 3 < inner_dim ? b_data[b_offset + (j + 3) * b_inc] : 1
                    );
                    if (_mm256_movemask_pd(_mm256_cmp_pd(b_vec, _mm256_setzero_pd(), _CMP_EQ_OQ)) != 0) {
                        throw std::runtime_error("Division by zero encountered.");
                    }
                    __m256d result_vec = _mm256_div_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result_data + result_offset + j, result_vec);
                }
            } else {
                for (size_t j = 0; j < inner_dim; ++j) {
                    T b_val = b_data[b_offset + j * b_inc];
                    if (b_val == 0) throw std::runtime_error("Division by zero encountered.");
                    result_data[result_offset + j] = a_data[a_offset + j * a_inc] / b_val;
                }
            }
        }

    return result;
}

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) const {
        // Get shapes of tensors a and b
        const std::vector<size_t>& a_shape = a.get_shape().dims();
        const std::vector<size_t>& b_shape = b.get_shape().dims();

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
        auto result = std::make_shared<CPUTensor<T>>(Shape{result_shape});

        // Compute the strides for each tensor
        const std::vector<size_t> a_strides = compute_strides(a_shape);
        const std::vector<size_t> b_strides = compute_strides(b_shape);
        const std::vector<size_t> result_strides = compute_strides(result_shape);

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

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        // Parallelize using OpenMP
#pragma omp parallel for
        for (size_t idx = 0; idx < num_matrices; ++idx) {
            std::vector<size_t> coords(depth);
            size_t temp = idx;
            for (int i = static_cast<int>(depth) - 1; i >= 0; --i) {
                coords[i] = temp % a_shape[i];
                temp /= a_shape[i];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t i = 0; i < depth; ++i) {
                a_offset += coords[i] * a_strides[i] * a_steps[i];
                b_offset += coords[i] * b_strides[i] * b_steps[i];
                result_offset += coords[i] * result_strides[i];
            }

            const T* a_sub = a_data + a_offset;
            const T* b_sub = b_data + b_offset;
            T* result_sub = result_data + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else {
                // Fallback for other types using vectorized operations
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; j += 4) {
                        __m128 sum = _mm_setzero_ps();
                        for (size_t l = 0; l < k; ++l) {
                            const __m128 a_val = _mm_set1_ps(a_sub[i * k + l]);
                            const __m128 b_val = _mm_loadu_ps(&b_sub[l * n + j]);
                            sum = _mm_add_ps(sum, _mm_mul_ps(a_val, b_val));
                        }
                        _mm_storeu_ps(&result_sub[i * n + j], sum);
                    }
                    // Handle remaining elements
                    for (size_t j = n - (n % 4); j < n; ++j) {
                        T sum = 0;
                        for (size_t l = 0; l < k; ++l) {
                            sum += a_sub[i * k + l] * b_sub[l * n + j];
                        }
                        result_sub[i * n + j] = sum;
                    }
                }
            }
        }

        return result;
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
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<size_t>>
        CPUTensorOps<T>::broadcastTensors(const Tensor<T>& a, const Tensor<T>& b) const{
        const std::vector<size_t>& shape1 = a.get_shape().dims();
        const std::vector<size_t>& shape2 = b.get_shape().dims();

        // 计算目标形状
        std::vector<size_t> targetShape;
        const int maxDims = static_cast<int>(std::max(shape1.size(), shape2.size()));
        targetShape.resize(maxDims);

        for (int i = 0; i < maxDims; ++i) {
            const size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
            const size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::runtime_error("Incompatible shapes for broadcasting");
            }
            targetShape[maxDims - 1 - i] = std::max(dim1, dim2);
        }

        // 计算输入向量的步长
        std::vector<int64_t> strides1(maxDims, 0), strides2(maxDims, 0);
        int64_t stride1 = 1, stride2 = 1;
        for (int i = static_cast<int>(shape1.size()) - 1; i >= 0; --i) {
            strides1[maxDims - shape1.size() + i] = shape1[i] == 1 ? 0 : stride1;
            stride1 *= static_cast<int64_t>(shape1[i]);
        }
        for (int i = static_cast<int>(shape2.size()) - 1; i >= 0; --i) {
            strides2[maxDims - shape2.size() + i] = shape2[i] == 1 ? 0 : stride2;
            stride2 *= static_cast<int64_t>(shape2[i]);
        }

        return {strides1, strides2, targetShape};
    }

    // Explicit template instantiation
    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
};


