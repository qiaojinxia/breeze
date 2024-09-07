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
    template<typename Dtype>
    class CPUTensorOps final: public TensorOps<Dtype>{
    public:

        void fill(Tensor<Dtype>& a, Dtype value) const override;

        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> add(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> subtract(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> divide(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> multiply(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> matmul(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const override;

        ~CPUTensorOps() override= default;

    };

}



#endif //CPUTENSOROPS_H
