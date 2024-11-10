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

        using EffectiveScalarT2 = std::conditional_t<std::is_same_v<ScalarT2, void>, ScalarT1, ScalarT2>;
        using ScalarResultType = binary_op_result_t<ScalarT1, EffectiveScalarT2>;

        virtual void fill(Tensor<ScalarT1>& a, ScalarT1 value) const = 0;
        virtual void arange(Tensor<ScalarT1>& a, ScalarT1 start, ScalarT1 step) const = 0;
        virtual void randn(Tensor<ScalarT1>& a) const = 0;

        // 矩阵乘法
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> matmul(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> sin(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> cos(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> tan(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> atan(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> log(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> log2(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> log10(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> exp(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> sqrt(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> rsqrt(const Tensor<ScalarT1>& a) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> abs(const Tensor<ScalarT1>& a) const = 0;

        // wise操作
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> pow(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> add(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> subtract(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> divide(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarResultType>> multiply(const Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;

        //reduce操作
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> sum(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> prod(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> max(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<i64>> arg_max(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> min(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<i64>> arg_min(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> mean(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> std(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim, bool unbiased) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> var(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim, bool unbiased) const = 0;
        [[nodiscard]] virtual std::shared_ptr<Tensor<ScalarT1>> norm(const Tensor<ScalarT1> &a, std::vector<index_t> &dims, int p, bool keep_dim) const = 0;

        virtual void add_inplace(Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        virtual void subtract_inplace(Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        virtual void multiply_inplace(Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;
        virtual void divide_inplace(Tensor<ScalarT1>& a, const Tensor<EffectiveScalarT2>& b) const = 0;


        virtual ~TensorOps() = default;
    };


}

#endif //TENSOROPS_H