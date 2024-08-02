//
// Created by mac on 2024/7/31.
//

#ifndef TENSOROPS_H
#define TENSOROPS_H
namespace Breeze {
    template<typename T>
    class Tensor;

    template<typename T>
    class TensorOps {
        public:
        // 基本操作接口
        virtual std::shared_ptr<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b) const = 0;
        virtual std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& a, const Tensor<T>& b) const = 0 ;
        virtual ~TensorOps() = default;
    };
}
#endif //TENSOROPS_H
