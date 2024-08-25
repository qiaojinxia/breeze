//
// Created by caomaobay on 2024/8/24.
//

#ifndef BREEZE_NEONOPS_H
#define BREEZE_NEONOPS_H
#include "SIMDOps.h"
#ifdef USE_NEON

namespace Breeze {

    template<typename T>
    class NEONOps final : public SIMDOps<T> {
    public:
        NEONOps(const NEONOps&) = delete;
        NEONOps& operator=(const NEONOps&) = delete;
        NEONOps() = default;
        ~NEONOps() override = default;

        static const NEONOps& getInstance() {
            static const NEONOps instance;
            return instance;
        }

        void multiply(T* destination, const T* a, const T* b,
                      size_t elements_size, int32_t dest_stride,
                      int32_t a_stride, int32_t b_stride) const override;

        void divide(T* destination, const T* a, const T* b,
            size_t elements_size, int32_t dest_stride, int32_t a_stride, int32_t b_stride) const override;

        void fill(T *data_ptr, T value, size_t aligned64_size) const override;

        void eq(const T* a, const T* b, bool* result, size_t elements_size) const override;
    };

} // namespace Breeze

#endif // USE_NEON

#endif // BREEZE_NEONOPS_H