//
// Created by caomaobay on 2024/8/24.
//

#ifndef SIMDOPS_H
#define SIMDOPS_H
#include <iostream>
namespace Breeze {
    template<typename T>
    class SIMDOps {
    public:
        virtual void multiply(T* destination, const T* a, const T* b,size_t elements_size, int32_t dest_stride,int32_t a_stride, int32_t b_stride) const = 0;
        virtual void divide(T* destination, const T* a, const T* b, size_t elements_size, int32_t dest_stride, int32_t a_stride, int32_t b_stride) const = 0;
        virtual void fill(T *data_ptr, T value, size_t aligned64_size) const = 0;
        virtual void eq(const T* a, const T* b, bool* result, size_t elements_size) const = 0;
        virtual ~SIMDOps() = default;
    };
}
#endif //SIMDOPS_H
