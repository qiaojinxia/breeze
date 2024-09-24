#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"
#include <random>


namespace Breeze {
    template<typename ScalarType>
    class CPUTensor final: public Tensor<ScalarType>, public std::enable_shared_from_this<CPUTensor<ScalarType>>  {
    public:
        ~CPUTensor() override;

        explicit CPUTensor(Shape shape);
        explicit CPUTensor(std::vector<index_t> shape);
        CPUTensor(std::initializer_list<index_t> shape);
        CPUTensor(Shape shape, ScalarType value);
        CPUTensor(const CPUTensor& other);
        CPUTensor(const CPUTensor& other, std::vector<index_t>&& shape);
        CPUTensor();
        CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape);
        CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data, index_t offset,
            std::vector<index_t> shape, std::vector<index_t> strides);
        CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data, std::vector<index_t> shape);

        ScalarType operator[](const std::string& index) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> sin() const override;
        [[nodiscard]] std::shared_ptr<TensorBase> cos() const override;
        [[nodiscard]] std::shared_ptr<TensorBase> tan() const override;
        [[nodiscard]] std::shared_ptr<TensorBase> atan() const override;

        [[nodiscard]] std::shared_ptr<TensorBase> sum(std::vector<index_t> dims) override;

        [[nodiscard]] std::shared_ptr<TensorBase> matmul(const TensorBase& rhs) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> pow(const TensorBase& rhs) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> operator+(const TensorBase& rhs) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> operator-(const TensorBase& rhs) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> operator*(const TensorBase& rhs) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> operator/(const TensorBase& rhs) const override;

        void operator+=(const TensorBase& rhs) override;
        void operator-=(const TensorBase& rhs) override;
        void operator*=(const TensorBase& rhs) override;
        void operator/=(const TensorBase& rhs) override;

        [[nodiscard]] std::shared_ptr<TensorBase> reshape(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> slice(const std::vector<std::string>& range_strings) override;
        [[nodiscard]] std::shared_ptr<TensorBase> view(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> unsqueeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> squeeze(index_t dim) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> squeeze() const override;
        [[nodiscard]] std::shared_ptr<TensorBase> expand(const std::vector<index_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> transpose(index_t dim0, index_t dim1) const override;
        [[nodiscard]] std::shared_ptr<TensorBase> permute(const std::vector<index_t>& dims) override;
        [[nodiscard]] std::shared_ptr<TensorBase> flatten() override;
        [[nodiscard]] std::shared_ptr<TensorBase> flatten(index_t start_dim, index_t end_dim) override;
        [[nodiscard]] std::shared_ptr<TensorBase> repeat(const std::vector<index_t>& repeats) const override;

        static std::shared_ptr<CPUTensor> cat(const std::vector<Tensor<ScalarType>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> stack(const std::vector<Tensor<ScalarType>*>& tensors, index_t dim);
        static std::shared_ptr<CPUTensor> randn(std::vector<index_t> shape);

        ScalarType* mutable_data() override;
        [[nodiscard]] const ScalarType* data() const override;
        [[nodiscard]] index_t align_size() const override;
        void set_initial_shape(Shape& shape) override;

        [[nodiscard]] std::shared_ptr<TensorBase> clone() const override;
        [[nodiscard]] std::shared_ptr<TensorBase> contiguous() override;

        [[nodiscard]] const ScalarType& at(const std::vector<index_t>& indices) const override;
        void set_value(const std::vector<index_t>& indices, ScalarType value) override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        [[nodiscard]] bool is_contiguous() const override;

        void fill(ScalarType value) override;
        void fill(const std::function<ScalarType(const std::vector<index_t>&)>& value_func) override;

        [[nodiscard]] std::vector<index_t> get_strides() const override;

        [[nodiscard]] index_t n_bytes() const override;

        static std::shared_ptr<CPUTensor> arange(ScalarType start, ScalarType end, ScalarType step);
        static std::shared_ptr<CPUTensor> scalar(ScalarType value);
        static std::shared_ptr<CPUTensor> vector(index_t size);

    protected:
        std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> memory_block_;
        static void combine_tensors_out(const std::vector<Tensor<ScalarType>*>& tensors, index_t dim, CPUTensor* result);
    };
} // namespace Breeze

#endif // CPU_TENSOR_H