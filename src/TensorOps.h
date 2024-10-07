#ifndef TENSOROPS_H
#define TENSOROPS_H

#include <memory>
#include <type_traits>
#include "ScalarType.h"

namespace Breeze {
    template<typename ScalarType>
    class Tensor;

    template<typename... ScalarTypes>
    class TensorOps {
    public:

        using ScalarT1 = std::tuple_element_t<0, std::tuple<ScalarTypes..., void, void>>;
        using ScalarT2 = std::tuple_element_t<1, std::tuple<ScalarTypes..., void, void>>;

        using effective_ScalarT2 = std::conditional_t<std::is_same_v<ScalarT2, void>, ScalarT1, ScalarT2>;

        using scalar_ResultTypeype = typename BinaryOpResultType<ScalarT1, effective_ScalarT2>::type;

        virtual void fill(Tensor<ScalarT1>& a, ScalarT1 value) const = 0;
        virtual void arange(Tensor<ScalarT1>& a, ScalarT1 start, ScalarT1 step) const = 0;
        virtual void randn(Tensor<ScalarT1>& a) const = 0;

        // 矩阵乘法
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> matmul(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> sin(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> cos(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> tan(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> atan(const Tensor<ScalarT1>& a) const = 0;

        // wise操作
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> pow(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> add(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> subtract(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> divide(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> multiply(const Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> sum(Tensor<ScalarT1>& a, std::vector<index_t>& dims) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> max(Tensor<ScalarT1>& a, std::vector<index_t>& dims) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<scalar_ResultTypeype>> mean(Tensor<ScalarT1>& a, std::vector<index_t>& dims) const = 0;

        virtual void add_inplace(Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        virtual void subtract_inplace(Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        virtual void multiply_inplace(Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;
        virtual void divide_inplace(Tensor<ScalarT1>& a, const Tensor<effective_ScalarT2>& b) const = 0;


        virtual ~TensorOps() = default;
    };


}

#endif //TENSOROPS_H