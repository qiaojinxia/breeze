// Created by mac on 2024/7/31.
//
#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__APPLE__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace Breeze {
    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::add(const Tensor<T>& a, const Tensor<T>& b) const {
        auto [a_strides, b_strides, target_shape] =
            this->calc_broadcast_shape(a.get_shape().dims(),b.get_shape().dims(),false);

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

        auto [a_strides, b_strides, target_shape] =
            this->calc_broadcast_shape(a.get_shape().dims(),b.get_shape().dims(),false);

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
    auto [a_strides, b_strides, target_shape] =
        this->calc_broadcast_shape(a.get_shape().dims(), b.get_shape().dims(), false);

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

            #if defined(__x86_64__) || defined(_M_X64)
            // x86/AVX2
            if constexpr (std::is_same_v<T, float>){

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
            }
    #elif defined(__APPLE__) && defined(__ARM_NEON)
            // Apple Silicon/NEON
            if constexpr (std::is_same_v<T, float>) {
                for (size_t j = 0; j < inner_dim; j += 4) {
                    float32x4_t a_vec = vld1q_f32(a_data + a_offset + j * a_inc);
                    float32x4_t b_vec = vld1q_f32(b_data + b_offset + j * b_inc);
                    float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
                    vst1q_f32(result_data + result_offset + j, result_vec);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                for (size_t j = 0; j < inner_dim; j += 2) {
                    float64x2_t a_vec = vld1q_f64(a_data + a_offset + j * a_inc);
                    float64x2_t b_vec = vld1q_f64(b_data + b_offset + j * b_inc);
                    float64x2_t result_vec = vmulq_f64(a_vec, b_vec);
                    vst1q_f64(result_data + result_offset + j, result_vec);
                }
            }
    #endif
            else {
                // Fallback scalar implementation
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
        auto [a_strides, b_strides, target_shape] =
            this->calc_broadcast_shape(a.get_shape().dims(), b.get_shape().dims(), false);

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

    #if defined(__x86_64__) || defined(_M_X64)
            // x86/AVX2
            if constexpr (std::is_same_v<T, float>){
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
                    const __m256 result_vec = _mm256_div_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result_data + result_offset + j, result_vec);
                }

            } else if constexpr (std::is_same_v<T, double>){
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
                    const __m256d result_vec = _mm256_div_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result_data + result_offset + j, result_vec);
                }
            }
    #elif defined(__APPLE__) && defined(__ARM_NEON)
            // Apple Silicon/NEON
            if constexpr (std::is_same_v<T, float>) {
                for (size_t j = 0; j < inner_dim; j += 4) {
                    const float32x4_t a_vec = vld1q_f32(a_data + a_offset + j * a_inc);
                    const float32x4_t b_vec = vld1q_f32(b_data + b_offset + j * b_inc);
                    if (vminvq_f32(b_vec) == 0.0f) {
                        throw std::runtime_error("Division by zero encountered.");
                    }
                    const float32x4_t result_vec = vdivq_f32(a_vec, b_vec);
                    vst1q_f32(result_data + result_offset + j, result_vec);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                for (size_t j = 0; j < inner_dim; j += 2) {
                    const float64x2_t a_vec = vld1q_f64(a_data + a_offset + j * a_inc);
                    const float64x2_t b_vec = vld1q_f64(b_data + b_offset + j * b_inc);
                    if (vminvq_f64(b_vec) == 0.0) {
                        throw std::runtime_error("Division by zero encountered.");
                    }
                    const float64x2_t result_vec = vdivq_f64(a_vec, b_vec);
                    vst1q_f64(result_data + result_offset + j, result_vec);
                }
            }
#endif
            // Fallback scalar implementation
            else {
                for (size_t j = 0; j < inner_dim; ++j) {
                    const T b_val = b_data[b_offset + j * b_inc];
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
        const std::vector<size_t> a_shape = a.get_shape().dims();
        const std::vector<size_t> b_shape = b.get_shape().dims();

        // Check for correct dimensions
        if (a_shape.size() < 2 || b_shape.size() < 2) {
            throw std::invalid_argument("Input tensors must have at least two dimensions for matrix multiplication.");
        }

        // Check if inner dimensions match
        if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
            throw std::invalid_argument("The inner dimensions must match for matrix multiplication.");
        }

        // Calculate the broadcast shape
        auto [a_strides, b_strides, result_shape] =
            this->calc_broadcast_shape(a_shape, b_shape, true);

        // Allocate result tensor
        auto result = std::make_shared<CPUTensor<T>>(Shape{result_shape});

        // Compute the strides for each tensor
        const std::vector<size_t> result_strides = result->get_strides();

        const size_t depth = result_shape.size() - 2;
        size_t m = a_shape[a_shape.size() - 2];
        size_t k = a_shape[a_shape.size() - 1];
        size_t n = b_shape[b_shape.size() - 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Calculate the number of 2D matrices
        size_t num_matrices = 1;
        for (size_t i = 0; i < depth; ++i) {
            num_matrices *= result_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->data();
        const auto& a_steps = a.get_steps();
        const auto& b_steps = b.get_steps();

        for (size_t idx = 0; idx < num_matrices; ++idx) {
            std::vector<size_t> coords(depth);
            size_t temp = idx;
            for (int i = static_cast<int>(depth) - 1; i >= 0; --i) {
                coords[i] = temp % result_shape[i];
                temp /= result_shape[i];
            }

            size_t a_offset = 0, b_offset = 0, result_offset = 0;
            for (size_t i = 0; i < depth; ++i) {
                a_offset += (coords[i] % a_shape[i]) * a_strides[i] * a_steps[i];
                b_offset += (coords[i] % b_shape[i]) * b_strides[i] * b_steps[i];
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
                // Fallback scalar implementation
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
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
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<size_t>>
    CPUTensorOps<T>::calc_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2,const bool matmul) const {
        // 计算目标形状
        std::vector<size_t> targetShape;
        const int maxDims = static_cast<int>(std::max(shape1.size(), shape2.size()));
        targetShape.resize(maxDims);
        if (matmul) {
            // 特殊处理矩阵乘法的情况
            if (shape1.size() < 2 || shape2.size() < 2) {
                throw std::runtime_error("For matmul, both shapes must have at least 2 dimensions");
            }

            // 处理除最后两个维度之外的部分
            for (int i = 0; i < maxDims - 2; ++i) {
                const size_t dim1 = i < shape1.size() - 2 ? shape1[i] : 1;
                const size_t dim2 = i < shape2.size() - 2 ? shape2[i] : 1;
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw std::runtime_error("Incompatible shapes for broadcasting in matmul");
                }
                targetShape[i] = std::max(dim1, dim2);
            }

            // 处理最后两个维度
            targetShape[maxDims - 2] = shape1[shape1.size() - 2];
            targetShape[maxDims - 1] = shape2[shape2.size() - 1];
        } else {
            for (int i = 0; i < maxDims; ++i) {
                const size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
                const size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw std::runtime_error("Incompatible shapes for broadcasting");
                }
                targetShape[maxDims - 1 - i] = std::max(dim1, dim2);
            }
        }

        // 计算输入向量的步长
        std::vector<int32_t> strides1(maxDims, 0), strides2(maxDims, 0);
        int32_t stride1 = 1, stride2 = 1;
        // 矩阵乘法的步长计算
        for (int i = static_cast<int>(shape1.size()) - 1; i >= 0; --i) {
            strides1[maxDims - shape1.size() + i] = shape1[i] == 1 ? 0 : stride1;
            stride1 *= static_cast<int32_t>(shape1[i]);
        }
        for (int i = static_cast<int>(shape2.size()) - 1; i >= 0; --i) {
            strides2[maxDims - shape2.size() + i] = shape2[i] == 1 ? 0 : stride2;
            stride2 *= static_cast<int32_t>(shape2[i]);
        }

        return {strides1, strides2, targetShape};
    }

    // Explicit template instantiation
    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
};


