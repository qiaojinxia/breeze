//
// Created by caomaobay on 2024/8/24.
//
#ifdef USE_NEON
#include "NEONOps.h"
#include <arm_neon.h>

namespace Breeze {

    template<>
    void NEONOps<float>::multiply(float* destination, const float* a, const float* b,
                              const int32_t num_elements, const int32_t dest_stride,
                              const int32_t a_stride, const int32_t b_stride) const {
        for (int32_t i = 0; i < num_elements; i += 4) {
            const float32x4_t a_vec = vld1q_f32(a + i * a_stride);
            const float32x4_t b_vec = vld1q_f32(b + i * b_stride);
            const float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
            vst1q_f32(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void NEONOps<double>::multiply(double* destination, const double* a, const double* b,
                                   const int32_t num_elements, const int32_t dest_stride,
                                   const int32_t a_stride, const int32_t b_stride) const {
        for (int32_t i = 0; i < num_elements; i += 2) {
            const float64x2_t a_vec = vld1q_f64(a + i * a_stride);
            const float64x2_t b_vec = vld1q_f64(b + i * b_stride);
            const float64x2_t result_vec = vmulq_f64(a_vec, b_vec);
            vst1q_f64(destination + i * dest_stride, result_vec);
        }
    }


    template<>
    void NEONOps<float>::divide(float* destination, const float* a, const float* b,
                                const int32_t num_elements, const int32_t dest_stride,
                                const int32_t a_stride, const int32_t b_stride) const {
        for (int32_t i = 0; i < num_elements; i += 4) {
            const float32x4_t a_vec = vld1q_f32(a + i * a_stride);
            const float32x4_t b_vec = vld1q_f32(b + i * b_stride);

            if (vminvq_f32(b_vec) == 0.0f) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const float32x4_t result_vec = vdivq_f32(a_vec, b_vec);
            vst1q_f32(destination + i * dest_stride, result_vec);
        }
    }

    template<>
    void NEONOps<double>::divide(double* destination, const double* a, const double* b,
                                 const int32_t num_elements, const int32_t dest_stride,
                                 const int32_t a_stride, const int32_t b_stride) const {
        for (int32_t i = 0; i < num_elements; i += 2) {
            const float64x2_t a_vec = vld1q_f64(a + i * a_stride);
            const float64x2_t b_vec = vld1q_f64(b + i * b_stride);

            if (vminvq_f64(b_vec) == 0.0) {
                throw std::runtime_error("Division by zero encountered.");
            }

            const float64x2_t result_vec = vdivq_f64(a_vec, b_vec);
            vst1q_f64(destination + i * dest_stride, result_vec);
        }
    }

    // Explicit instantiation for float and double
    template class NEONOps<float>;
    template class NEONOps<double>;
}
#endif // USE_NEON