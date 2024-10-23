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
        using ScalarT2 = typename BaseOps::EffectiveScalarT2;
        using ScalarResult = typename BaseOps::ScalarResultType;

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

        [[nodiscard]]  std::shared_ptr<Tensor<ScalarT1>> sum(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<ScalarT1>> max(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<ScalarT1>> min(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<ScalarT1>> mean(const Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim) const override;
        [[nodiscard]]  std::shared_ptr<Tensor<ScalarT1>> std(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim, bool unbiased) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> var(Tensor<ScalarT1>& a, std::vector<index_t>& dims, bool keep_dim, bool unbiased) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> sin(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> cos(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> tan(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> atan(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> log(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> log2(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> log10(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> exp(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> sqrt(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> rsqrt(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> abs(const Tensor<ScalarT1>& a) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarT1>> norm(const Tensor<ScalarT1> &a, std::vector<index_t> &dims, int p, bool keep_dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> pow(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> add(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> subtract(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> divide(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> multiply(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        [[nodiscard]] std::shared_ptr<Tensor<ScalarResult>> matmul(const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;

        void add_inplace(Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        void subtract_inplace(Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        void multiply_inplace(Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;
        void divide_inplace(Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b) const override;

    private:
        // 私有构造函数
        CPUTensorOps() = default;
        // 虚析构函数
        ~CPUTensorOps() override = default;
    };

}

#endif //CPUTENSOROPS_H