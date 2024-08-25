//
// Created by caomaobay on 2024/8/24.
//
#ifdef USE_NEON
#include "NEONOps.h"
#include <arm_neon.h>
#include <omp.h>
#include <cassert>
namespace Breeze {

    template<>
    void NEONOps<float>::multiply(float* destination, const float* a, const float* b,
                              const size_t elements_size, const int32_t dest_stride,
                              const int32_t a_stride, const int32_t b_stride) const {
        size_t i = 0;
        for (; i < elements_size - 3; i += 4) {
            const float32x4_t a_vec = vld1q_f32(a + i * a_stride);
            const float32x4_t b_vec = vld1q_f32(b + i * b_stride);
            const float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
            vst1q_f32(destination + i * dest_stride, result_vec);
        }
        for (; i < elements_size; i++) {  // 处理剩余的元素
            destination[i * dest_stride] = a[i * a_stride] * b[i * b_stride];
        }
    }

    template<>
    void NEONOps<double>::multiply(double* destination, const double* a, const double* b,
                                   const size_t elements_size, const int32_t dest_stride,
                                   const int32_t a_stride, const int32_t b_stride) const {
        size_t i = 0;
        for (; i < elements_size; i += 2) {
            const float64x2_t a_vec = vld1q_f64(a + i * a_stride);
            const float64x2_t b_vec = vld1q_f64(b + i * b_stride);
            const float64x2_t result_vec = vmulq_f64(a_vec, b_vec);
            vst1q_f64(destination + i * dest_stride, result_vec);
        }
        for (; i < elements_size; i++) {  // 处理剩余的元素
            destination[i * dest_stride] = a[i * a_stride] * b[i * b_stride];
        }
    }


    template<>
    void NEONOps<float>::divide(float* destination, const float* a, const float* b,
                                const size_t elements_size, const int32_t dest_stride,
                                const int32_t a_stride, const int32_t b_stride) const {
        size_t i = 0;
        for (; i < elements_size; i += 4) {
            const float32x4_t a_vec = vld1q_f32(a + i * a_stride);
            const float32x4_t b_vec = vld1q_f32(b + i * b_stride);

            if (vminvq_f32(b_vec) == 0.0f) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const float32x4_t result_vec = vdivq_f32(a_vec, b_vec);
            vst1q_f32(destination + i * dest_stride, result_vec);
        }
        for (; i < elements_size; i++) {
            if (b[i * b_stride] == 0.0f) {
                throw std::runtime_error("Division by zero encountered.");
            }
            destination[i * dest_stride] = a[i * a_stride] / b[i * b_stride];
        }
    }

    template<>
    void NEONOps<double>::divide(double* destination, const double* a, const double* b,
                                 const size_t elements_size, const int32_t dest_stride,
                                 const int32_t a_stride, const int32_t b_stride) const {
        size_t i = 0;
        for (; i < elements_size; i += 2) {
            const float64x2_t a_vec = vld1q_f64(a + i * a_stride);
            const float64x2_t b_vec = vld1q_f64(b + i * b_stride);

            if (vminvq_f64(b_vec) == 0.0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const float64x2_t result_vec = vdivq_f64(a_vec, b_vec);
            vst1q_f64(destination + i * dest_stride, result_vec);
        }
        for (; i < elements_size; i++) {
            if (b[i * b_stride] == 0.0f) {
                throw std::runtime_error("Division by zero encountered.");
            }
            destination[i * dest_stride] = a[i * a_stride] / b[i * b_stride];
        }
    }


    template<>
    void NEONOps<float>::eq(const float* a, const float* b, bool* result,const size_t elements_size) const {

        // 每次处理8个float（两个128位向量）
        for (size_t i = 0; i<= elements_size; i += 8) {
            const float32x4_t va1 = vld1q_f32(a + i);
            const float32x4_t vb1 = vld1q_f32(b + i);
            const float32x4_t va2 = vld1q_f32(a + i + 4);
            const float32x4_t vb2 = vld1q_f32(b + i + 4);

            const uint32x4_t vcmp1 = vceqq_f32(va1, vb1);
            const uint32x4_t vcmp2 = vceqq_f32(va2, vb2);

            uint64_t mask = vgetq_lane_u64(vreinterpretq_u64_u32(vcmp1), 0) &
                            vgetq_lane_u64(vreinterpretq_u64_u32(vcmp1), 1) &
                            vgetq_lane_u64(vreinterpretq_u64_u32(vcmp2), 0) &
                            vgetq_lane_u64(vreinterpretq_u64_u32(vcmp2), 1);

            memcpy(result + i, &mask, sizeof(uint64_t));
        }
    }

    template<>
    void NEONOps<double>::eq(const double* a, const double* b, bool* result, const size_t elements_size) const {

        for (size_t i = 0; i < elements_size; i += 4) {
            const float64x2_t va1 = vld1q_f64(a + i);
            const float64x2_t vb1 = vld1q_f64(b + i);
            const float64x2_t va2 = vld1q_f64(a + i + 2);
            const float64x2_t vb2 = vld1q_f64(b + i + 2);

            const uint64x2_t vcmp1 = vceqq_f64(va1, vb1);
            const uint64x2_t vcmp2 = vceqq_f64(va2, vb2);

            uint64_t mask = vgetq_lane_u64(vcmp1, 0) & vgetq_lane_u64(vcmp1, 1) &
                            vgetq_lane_u64(vcmp2, 0) & vgetq_lane_u64(vcmp2, 1);

            memcpy(result + i, &mask, sizeof(uint32_t));
        }
    }

    template<>
    void NEONOps<float>::fill(float *data_ptr, const float value, const size_t aligned64_size) const {
            assert(aligned64_size % 4 == 0 && "Size must be a multiple of 4 for 64-bit alignment");
            // 使用NEON指令集
            const float32x4_t wide_val = vdupq_n_f32(value);
            // 每次处理4个float
            for (size_t i = 0; i < aligned64_size; i += 4) {
                __builtin_prefetch(data_ptr + i + 16, 1);
                vst1q_f32(data_ptr + i, wide_val);
            }
        }

    template<>
    void NEONOps<double>::fill(double *data_ptr, const double value, const size_t aligned64_size) const  {
        assert(aligned64_size % 4 == 0 && "Size must be a multiple of 4 for 64-bit alignment");
        // 使用NEON指令集
        const float64x2_t wide_val = vdupq_n_f64(value);
        // #pragma omp parallel for
        for (size_t i = 0; i< aligned64_size; i += 2) {
            // 预取下一个缓存行
            __builtin_prefetch(data_ptr + i + 16, 1);
            vst1q_f64(data_ptr + i, wide_val);
        }
    }

    // Explicit instantiation for float and double
    template class NEONOps<float>;
    template class NEONOps<double>;
}
#endif // USE_NEON