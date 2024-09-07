#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"
#include <random>

namespace Breeze {
    template<typename Dtype>
    class CPUTensor final : public Tensor<Dtype>, public std::enable_shared_from_this<CPUTensor<Dtype>>  {
    public:
        ~CPUTensor() override;

        explicit CPUTensor(Shape shape);
        explicit CPUTensor(std::vector<index_t> shape);
        CPUTensor(std::initializer_list<index_t> shape);
        CPUTensor(Shape shape, Dtype value);
        CPUTensor(const CPUTensor& other);
        CPUTensor(const CPUTensor& other, std::vector<index_t>&& shape);
        CPUTensor();
        CPUTensor(std::shared_ptr<TensorStorage<Dtype, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape);
        CPUTensor(std::shared_ptr<TensorStorage<Dtype, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape, std::vector<index_t> strides);
        CPUTensor(std::shared_ptr<TensorStorage<Dtype, CPUDevice>> data, std::vector<index_t> shape);

        Dtype operator[](const std::string& index) const override;
        std::shared_ptr<Tensor<Dtype>> operator+(const Tensor<Dtype>& rhs) const override;
        std::shared_ptr<Tensor<Dtype>> operator-(const Tensor<Dtype>& rhs) const override;
        std::shared_ptr<Tensor<Dtype>> operator*(const Tensor<Dtype>& rhs) const override;
        std::shared_ptr<Tensor<Dtype>> operator/(const Tensor<Dtype>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> matmul(const Tensor<Dtype>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> reshape(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> slice(const std::vector<std::string>& range_strings) override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> view(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> unsqueeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> squeeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> squeeze() const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> expand(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> transpose(index_t dim0, index_t dim1) const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> permute(const std::vector<index_t>& dims) override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> flatten() override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> flatten(index_t start_dim, index_t end_dim) override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> repeat(const std::vector<index_t>& repeats) const override;

        static std::shared_ptr<CPUTensor> cat(const std::vector<Tensor<Dtype>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> stack(const std::vector<Tensor<Dtype>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> randn(std::vector<index_t> shape);
        static std::shared_ptr<CPUTensor> randn(std::vector<index_t>& shape, std::default_random_engine& generator);
        static bool isSafeToModify(std::shared_ptr<Tensor<Dtype>> tensor);

        Dtype* mutable_data() override;
        [[nodiscard]] const Dtype* data() const override;
        [[nodiscard]] index_t align_size() const override;
        void set_initial_shape(Shape& shape) override;

        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> clone() const override;
        [[nodiscard]] std::shared_ptr<Tensor<Dtype>> contiguous() override;

        [[nodiscard]] const Dtype& at(const std::vector<index_t>& indices) const override;
        void set_value(const std::vector<index_t>& indices, Dtype value) override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        [[nodiscard]] bool is_contiguous() const override;

        void fill(Dtype value) override;
        void fill(const std::function<Dtype(const std::vector<index_t>&)>& value_func) override;

        [[nodiscard]] std::vector<index_t> get_strides() const override;

        [[nodiscard]] index_t n_bytes() const override;

        static std::shared_ptr<CPUTensor> arange(Dtype start, Dtype end, Dtype step);
        static std::shared_ptr<CPUTensor> scalar(Dtype value);
        static std::shared_ptr<CPUTensor> vector(index_t size);

    private:
        std::shared_ptr<TensorStorage<Dtype, CPUDevice>> memory_block_;
        static void combine_tensors_out(const std::vector<Tensor<Dtype>*>& tensors, index_t dim, CPUTensor* result);
    };
} // namespace Breeze

#endif // CPU_TENSOR_H