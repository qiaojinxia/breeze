#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"
#include <cblas.h>


namespace Breeze {

    template<typename T>
    class CPUTensor final : public Tensor<T>, public std::enable_shared_from_this<CPUTensor<T>>  {
    public:
        ~CPUTensor() override ;
        explicit CPUTensor(Shape shape);
        explicit CPUTensor(std::vector<size_t> shape_size);
        CPUTensor(std::initializer_list<size_t> shape_size);
        CPUTensor(Shape shape, T value);
        CPUTensor(const CPUTensor& other);
        CPUTensor();
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, size_t offset,
            std::vector<size_t> shape_size);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, size_t offset,
            std::vector<size_t> shape_size, std::vector<int32_t> steps);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, size_t offset,
            std::vector<size_t> shape_size, std::vector<int32_t> steps, std::vector<size_t> strides);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, std::vector<size_t> shape_size);

        std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& rhs) const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> reshape(const std::vector<int32_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> slice(const std::vector<std::string>& range_strings) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> view(const std::vector<int32_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> unsqueeze(int32_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> squeeze(int32_t dim) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> expand(const std::vector<int32_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> transpose(int32_t dim0, int32_t dim1) const override;

        static std::shared_ptr<CPUTensor> cat(const std::vector<Tensor<T>*>& tensors, int32_t dim);

        T* data() override;
        [[nodiscard]] const T* data() const override;

        [[nodiscard]] std::shared_ptr<Tensor<T>> clone() const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> contiguous() override;

        [[nodiscard]] const T& at(const std::vector<size_t>& indices) const override;
        void set_value(const std::vector<size_t>& indices, T value) override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        [[nodiscard]] bool is_contiguous() const override;

        void fill(T value) override;
        void fill(const std::function<T(const std::vector<size_t>&)>& value_func) override;

        [[nodiscard]] std::vector<int32_t> get_steps() const override;
        [[nodiscard]] std::vector<size_t> get_strides() const override;
        [[nodiscard]] size_t n_bytes() const override;

        static std::shared_ptr<CPUTensor> arrange(T begin, T end, T step);
        static std::shared_ptr<CPUTensor> scalar();
        static std::shared_ptr<CPUTensor> vector(size_t size);

    private:
        std::shared_ptr<TensorStorage<T, CPUDevice>> memory_block_;
        size_t offset_ = 0;
        std::vector<int32_t> steps_;
        std::vector<size_t> strides_;
    };
} // namespace Breeze

#endif // CPU_TENSOR_H