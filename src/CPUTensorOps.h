//
// Created by mac on 2024/7/31.
//

#ifndef CPUTENSOROPS_H
#define CPUTENSOROPS_H
#include <vector>
#include "TensorOps.h"

namespace Breeze {
    enum class TensorOpType {
        Add,
        Subtract,
        Multiply,
        Divide
    };
    template<typename T>
    class CPUTensorOps final: public TensorOps<T>{
    public:

        [[nodiscard]] std::shared_ptr<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> subtract(const Tensor<T>& a, const Tensor<T>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> divide(const Tensor<T>& a, const Tensor<T>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> multiply(const Tensor<T>& a, const Tensor<T>& b) const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& a, const Tensor<T>& b) const override;

        [[nodiscard]] std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<size_t>>
            calc_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2,bool matmul) const override;

        ~CPUTensorOps() override= default;

    };

}



#endif //CPUTENSOROPS_H
