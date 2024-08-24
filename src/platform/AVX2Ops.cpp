//
// Created by caomaobay on 2024/8/24.
//
#ifdef USE_AVX2
#include "AVX2Ops.h"
#include <immintrin.h>
namespace Breeze {

    template<>
    void AVX2Ops<float>::multiply(float* destination, const float* a, const float* b,
                                  const int32_t num_elements, const int32_t dest_stride,
                                  const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 8;  // 256 bits / 32 bits
        const size_t aligned_size = (num_elements / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256 a_vec = _mm256_load_ps(a + i * a_stride);
            const __m256 b_vec = _mm256_load_ps(b + i * b_stride);
            __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
            _mm256_store_ps(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void AVX2Ops<double>::multiply(double* destination, const double* a, const double* b,
                                   const int32_t num_elements, const int32_t dest_stride,
                                   const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 4;  // 256 bits / 64 bits
        const size_t aligned_size = (num_elements / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256d a_vec = _mm256_load_pd(a + i * a_stride);
            const __m256d b_vec = _mm256_load_pd(b + i * b_stride);
            __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_store_pd(destination + i * dest_stride, result_vec);
        }

    }


    template<>
    void AVX2Ops<float>::divide(float* destination, const float* a, const float* b,
                                const int32_t num_elements, const int32_t dest_stride,
                                const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 8;  // 256 bits / 32 bits
        const size_t aligned_size = (num_elements / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256 a_vec = _mm256_load_ps(a + i * a_stride);
            const __m256 b_vec = _mm256_load_ps(b + i * b_stride);

            // Check for division by zero
            __m256 zero_mask = _mm256_cmp_ps(b_vec, _mm256_setzero_ps(), _CMP_EQ_OQ);
            if (_mm256_movemask_ps(zero_mask) != 0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            __m256 reciprocal = _mm256_rcp_ps(b_vec);
            __m256 result_vec = _mm256_mul_ps(a_vec, reciprocal);
            _mm256_store_ps(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void AVX2Ops<double>::divide(double* destination, const double* a, const double* b,
                                 const int32_t num_elements, const int32_t dest_stride,
                                 const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 4;  // 256 bits / 64 bits
        const size_t aligned_size = (num_elements / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256d a_vec = _mm256_load_pd(a + i * a_stride);
            const __m256d b_vec = _mm256_load_pd(b + i * b_stride);

            // Check for division by zero
            __m256d zero_mask = _mm256_cmp_pd(b_vec, _mm256_setzero_pd(), _CMP_EQ_OQ);
            if (_mm256_movemask_pd(zero_mask) != 0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            __m256d result_vec = _mm256_div_pd(a_vec, b_vec);
            _mm256_store_pd(destination + i * dest_stride, result_vec);
        }

    }


    // Explicit instantiation
    template class AVX2Ops<float>;
    template class AVX2Ops<double>;
}
#endif // USE_AVX2