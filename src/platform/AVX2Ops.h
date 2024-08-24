//
// Created by caomaobay on 2024/8/24.
//

#ifndef BREEZE_AVX2OPS_H
#define BREEZE_AVX2OPS_H

#include "SIMDOps.h"

#ifdef USE_AVX2

namespace Breeze {

    template<typename T>
    class AVX2Ops final : public SIMDOps<T> {
    public:
        AVX2Ops() = default;
        ~AVX2Ops() override = default;
        AVX2Ops(const AVX2Ops&) = delete;
        AVX2Ops& operator=(const AVX2Ops&) = delete;

        static const AVX2Ops& getInstance() {
            static const AVX2Ops instance;
            return instance;
        }

        void multiply(T* destination, const T* a, const T* b,
                      int32_t num_elements, int32_t dest_stride,
                      int32_t a_stride, int32_t b_stride) const override;

        void divide(T* destination, const T* a, const T* b,
                   int32_t num_elements, int32_t dest_stride, int32_t a_stride, int32_t b_stride) const override;
    };

} // namespace Breeze

#endif // USE_AVX2

#endif // BREEZE_AVX2OPS_H