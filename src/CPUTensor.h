#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"
#include <cblas.h>
namespace Breeze {

    // 定义特殊值
    template<typename T>
    class CPUTensor final: public Tensor<T> {
    public:
        explicit CPUTensor(const Shape& _shape);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, size_t offset,
            const std::vector<size_t>&& shape_size, std::vector<int32_t>&& steps);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, size_t offset,
            const std::vector<size_t>&& shape_size, std::vector<int32_t>&& steps, std::vector<size_t>&& strides);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, const std::vector<size_t>&& shape_size);

        ~CPUTensor() override ;

        std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) const override;

        std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& rhs) const override;

        void resize(const Shape& new_shape) override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> reshape(const std::vector<int32_t>& new_shape) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> slice(const std::vector<std::pair<int32_t, int32_t>>& ranges) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> slice(const std::vector<std::tuple<int32_t, int32_t, int32_t>>& ranges) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> view(const std::vector<int32_t>& new_shape) const override;
        void expand(const Shape&& new_shape) override;

        static std::shared_ptr<CPUTensor> cat(const std::vector<CPUTensor*>& tensors, int32_t dim);

        T* data() override;
        const T* data() const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> clone() const override;

        [[nodiscard]] const T& at(const std::vector<size_t>& indices) const override;
        void set_value(const std::vector<size_t>& indices,T value) override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        [[nodiscard]] bool is_contiguous() const override;

        void fill(T value) override;

        void fill(const std::function<T(const std::vector<size_t>&)>& value_func) override;

        [[nodiscard]] std::vector<int32_t> get_steps() const override;

        void setTensorStorage(std::shared_ptr<TensorStorage<T, CPUDevice>> new_block,Shape&& n_shape);

        [[nodiscard]] std::vector<size_t> get_strides() const override;

        static std::shared_ptr<CPUTensor> arrange(T  begin,T end,T step);
    private:
        std::shared_ptr<TensorStorage<T, CPUDevice>> memory_block;
        size_t offset_ = 0;
        std::vector<int32_t> steps_;
        std::vector<size_t> strides_;
    };

} // namespace Breeze

#endif // CPU_TENSOR_H