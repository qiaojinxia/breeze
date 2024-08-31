//
// Created by caomaobay on 2024/8/24.
//
#ifdef USE_AVX2
#include "AVX2Ops.h"
#include <immintrin.h>
#include <cassert>
namespace Breeze {

    template<>
    void AVX2Ops<float>::multiply(float* destination, const float* a, const float* b,
                                  const size_t elements_size, const int32_t dest_stride,
                                  const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 8;  // 256 bits / 32 bits
        const size_t aligned_size = (elements_size / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256 a_vec = _mm256_load_ps(a + i * a_stride);
            const __m256 b_vec = _mm256_load_ps(b + i * b_stride);
            const __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
            _mm256_store_ps(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void AVX2Ops<double>::multiply(double* destination, const double* a, const double* b,
                                   const size_t elements_size, const int32_t dest_stride,
                                   const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 4;  // 256 bits / 64 bits
        const size_t aligned_size = (elements_size / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256d a_vec = _mm256_load_pd(a + i * a_stride);
            const __m256d b_vec = _mm256_load_pd(b + i * b_stride);
            const __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_store_pd(destination + i * dest_stride, result_vec);
        }

    }


    template<>
    void AVX2Ops<float>::divide(float* destination, const float* a, const float* b,
                                const size_t elements_size, const int32_t dest_stride,
                                const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 8;  // 256 bits / 32 bits
        const size_t aligned_size = (elements_size / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256 a_vec = _mm256_load_ps(a + i * a_stride);
            const __m256 b_vec = _mm256_load_ps(b + i * b_stride);

            // Check for division by zero
            if (const __m256 zero_mask = _mm256_cmp_ps(b_vec, _mm256_setzero_ps(), _CMP_EQ_OQ);
                _mm256_movemask_ps(zero_mask) != 0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const __m256 reciprocal = _mm256_rcp_ps(b_vec);
            const __m256 result_vec = _mm256_mul_ps(a_vec, reciprocal);
            _mm256_store_ps(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void AVX2Ops<double>::divide(double* destination, const double* a, const double* b,
                                 const size_t elements_size, const int32_t dest_stride,
                                 const int32_t a_stride, const int32_t b_stride) const {
        constexpr size_t vector_size = 4;  // 256 bits / 64 bits
        const size_t aligned_size = (elements_size / vector_size) * vector_size;

        for (int32_t i = 0; i < aligned_size; i += vector_size) {
            const __m256d a_vec = _mm256_load_pd(a + i * a_stride);
            const __m256d b_vec = _mm256_load_pd(b + i * b_stride);

            // Check for division by zero
            if (const __m256d zero_mask = _mm256_cmp_pd(b_vec, _mm256_setzero_pd(), _CMP_EQ_OQ);
                _mm256_movemask_pd(zero_mask) != 0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const __m256d result_vec = _mm256_div_pd(a_vec, b_vec);
            _mm256_store_pd(destination + i * dest_stride, result_vec);
        }

    }

    template<>
    void AVX2Ops<float>::fill(float *data_ptr, const float value, const size_t aligned64_size) const {
            assert(aligned64_size % 8 == 0 && "Size must be a multiple of 8 for 256-bit alignment");
            const __m256 wide_val = _mm256_set1_ps(value);  // 复制 value 到 AVX2 寄存器的所有位置
            for (size_t i = 0; i < aligned64_size; i += 8) {
                _mm256_stream_ps(data_ptr + i, wide_val);  // 使用非临时存储写入数据
            }
    }

    template<>
    void AVX2Ops<double>::fill(double *data_ptr, const double value, const size_t aligned64_size) const {
        assert(aligned64_size % 4 == 0 && "Size must be a multiple of 4 for 256-bit alignment");
        const __m256d wide_val = _mm256_set1_pd(value);  // 复制 value 到 AVX2 寄存器的所有位置
        for (size_t i = 0; i < aligned64_size; i += 4) {
            _mm256_stream_pd(data_ptr + i, wide_val);  // 使用非临时存储写入数据
        }
    }



    template<>
    void AVX2Ops<float>::eq(const float* a, const float* b, bool* result,const size_t elements_size) const {
        //todo
    }

    template<>
    void AVX2Ops<double>::eq(const double* a, const double* b, bool* result, const size_t elements_size) const {
        //todo
    }

    // Explicit instantiation
    template class AVX2Ops<float>;
    template class AVX2Ops<double>;
}
#endif // USE_AVX2