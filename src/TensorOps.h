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
        //矩阵乘法
        virtual std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& a, const Tensor<T>& b) const = 0 ;

        //wise操作
        virtual std::shared_ptr<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b) const = 0;
        virtual std::shared_ptr<Tensor<T>> subtract(const Tensor<T>& a, const Tensor<T>& b) const = 0;
        virtual std::shared_ptr<Tensor<T>> divide(const Tensor<T>& a, const Tensor<T>& b) const = 0 ;
        virtual std::shared_ptr<Tensor<T>> multiply(const Tensor<T>& a, const Tensor<T>& b) const = 0;

        [[nodiscard]] virtual std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<size_t>>
            calc_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2, bool matmul) const = 0;


        virtual ~TensorOps() = default;
    };
}
#endif //TENSOROPS_H
