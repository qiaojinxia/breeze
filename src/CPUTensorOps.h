//
// Created by mac on 2024/7/31.
//

#ifndef CPUTENSOROPS_H
#define CPUTENSOROPS_H
#include <vector>
#include "TensorOps.h"
namespace Breeze {
    template<typename T>
    class CPUTensorOps final: public TensorOps<T>{
    public:
        [[nodiscard]] std::shared_ptr<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& a, const Tensor<T>& b) const override;
        ~CPUTensorOps() override= default;
    private:
        void multiply_non_recursive(const T* a, const T* b, T* result, const std::vector<size_t>& a_shape,
                                             const std::vector<size_t>& b_shape,
                                             const std::vector<size_t>& a_strides, const std::vector<size_t>& b_strides,
                                             const std::vector<size_t>& result_strides) const;

        [[nodiscard]]  static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
    };

}



#endif //CPUTENSOROPS_H
