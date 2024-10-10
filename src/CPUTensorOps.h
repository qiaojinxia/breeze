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

    template<typename... ScalarTypes>
    class CPUTensorOps final : public TensorOps<ScalarTypes...> {
    public:
        using BaseOps = TensorOps<ScalarTypes...>;
        using ScalarT1 = typename BaseOps::ScalarT1;
        using ScalarT2 = typename BaseOps::effective_ScalarT2;
        using scalar_result = typename BaseOps::scalar_ResultTypeype;

        void fill(Tensor<ScalarT1>& a, ScalarT1 value) const override;
        void arange(Tensor<ScalarT1>& a, ScalarT1 start, ScalarT1 step) const override;
        void randn(Tensor<ScalarT1>& a) const override;

        // 删除拷贝构造函数和赋值操作符
        CPUTensorOps(const CPUTensorOps&) = delete;
        CPUTensorOps& operator=(const CPUTensorOps&) = delete;

        // 静态方法获取单例实例
        static const CPUTensorOps& getInstance() {
            static CPUTensorOps instance;
            return instance;
        }


        [[nodiscard]]  std::shared_ptr<Tensor<scalar_result>> sum(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keepdim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<scalar_result>> max(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keepdim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<scalar_result>> min(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keepdim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<scalar_result>> mean(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keepdim) const override;

        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> sin(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> cos(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> tan(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> atan(const Tensor<ScalarT1>& a) const override;

        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> pow(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> add(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> subtract(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> divide(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> multiply(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<scalar_result>> matmul(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;

        void add_inplace(Tensor<ScalarT1>& a,  const Tensor<ScalarT2>& b) const override;
        void subtract_inplace(Tensor<ScalarT1>& a,  const Tensor<ScalarT2>& b) const override;
        void multiply_inplace(Tensor<ScalarT1>& a,  const Tensor<ScalarT2>& b) const override;
        void divide_inplace(Tensor<ScalarT1>& a,  const Tensor<ScalarT2>& b) const override;


    private:
        // 私有构造函数
        CPUTensorOps() = default;
        // 虚析构函数
        ~CPUTensorOps() override = default;
    };

}

#endif //CPUTENSOROPS_H