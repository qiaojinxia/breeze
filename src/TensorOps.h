//
// Created by mac on 2024/7/31.
//

#ifndef TENSOROPS_H
#define TENSOROPS_H
namespace Breeze {
    template<typename Dtype>
    class Tensor;

    template<typename Dtype>
    class TensorOps {
        public:

        virtual void fill(Tensor<Dtype>& a, Dtype value) const = 0;
        //矩阵乘法
        virtual std::shared_ptr<Tensor<Dtype>> matmul(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const = 0;

        //wise操作
        virtual std::shared_ptr<Tensor<Dtype>> add(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const = 0;
        virtual std::shared_ptr<Tensor<Dtype>> subtract(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const = 0;
        virtual std::shared_ptr<Tensor<Dtype>> divide(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const = 0;
        virtual std::shared_ptr<Tensor<Dtype>> multiply(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const = 0;


        virtual ~TensorOps() = default;
    };
}
#endif //TENSOROPS_H
