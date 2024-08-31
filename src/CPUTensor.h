#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"
#include <random>

namespace Breeze {
    template<typename T>
    class CPUTensor final : public Tensor<T>, public std::enable_shared_from_this<CPUTensor<T>>  {
    public:
        ~CPUTensor() override;

        explicit CPUTensor(Shape shape);
        explicit CPUTensor(std::vector<index_t> shape);
        CPUTensor(std::initializer_list<index_t> shape);
        CPUTensor(Shape shape, T value);
        CPUTensor(const CPUTensor& other);
        CPUTensor(const CPUTensor& other, std::vector<index_t>&& shape);
        CPUTensor();
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape, std::vector<index_t> strides);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, std::vector<index_t> shape);

        T operator[](const std::string& index) const override;
        std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> reshape(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> slice(const std::vector<std::string>& range_strings) override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> view(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> unsqueeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> squeeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> squeeze() const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> expand(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> transpose(index_t dim0, index_t dim1) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> permute(const std::vector<index_t>& dims) override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> flatten() override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> flatten(index_t start_dim, index_t end_dim) override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> repeat(const std::vector<index_t>& repeats) const override;

        static std::shared_ptr<CPUTensor> cat(const std::vector<Tensor<T>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> stack(const std::vector<Tensor<T>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> randn(std::vector<index_t> shape);
        static std::shared_ptr<CPUTensor> randn(std::vector<index_t>& shape, std::default_random_engine& generator);
        static bool isSafeToModify(std::shared_ptr<Tensor<T>> tensor);

        T* mutable_data() override;
        [[nodiscard]] const T* data() const override;
        [[nodiscard]] index_t align_size() const override;
        void set_initial_shape(Shape& shape) override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> clone() const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> contiguous() override;

        [[nodiscard]] const T& at(const std::vector<index_t>& indices) const override;
        void set_value(const std::vector<index_t>& indices, T value) override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        [[nodiscard]] bool is_contiguous() const override;

        void fill(T value) override;
        void fill(const std::function<T(const std::vector<index_t>&)>& value_func) override;

        [[nodiscard]] std::vector<index_t> get_strides() const override;

        [[nodiscard]] index_t n_bytes() const override;

        static std::shared_ptr<CPUTensor> arange(T start, T end, T step);
        static std::shared_ptr<CPUTensor> scalar(T value);
        static std::shared_ptr<CPUTensor> vector(index_t size);

    private:
        std::shared_ptr<TensorStorage<T, CPUDevice>> memory_block_;
        static void combine_tensors_out(const std::vector<Tensor<T>*>& tensors, index_t dim, CPUTensor* result);
    };
} // namespace Breeze

#endif // CPU_TENSOR_H