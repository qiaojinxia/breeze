#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include "Tensor.h"
#include "TensorStorage.h"

namespace Breeze {

    template<typename T>
    class CPUTensor final: public Tensor<T> {
    public:
        explicit CPUTensor(const Shape& _shape);
        CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data,size_t offset,
            const std::vector<size_t>&& shape_size, const std::vector<size_t>&& strides);

        ~CPUTensor() override ;

        std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) const override;

        std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& rhs) const override;

        void broadcast(Tensor<T>& rhs) override;

        void resize(const Shape& new_shape) override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const override;
        [[nodiscard]] std::shared_ptr<Tensor<T>> view(const Shape& new_shape) const override;


        T* data() override;
        const T* data() const override;

        void to_cpu() override;
        void to_gpu() override;

        void print(std::ostream& os) const override;

        void fill(T value) override;

        void setTensorStorage(std::shared_ptr<TensorStorage<T, CPUDevice>> new_block,Shape&& n_shape);
    private:
        std::shared_ptr<TensorStorage<T, CPUDevice>> memory_block;
        size_t offset_ = 0;
        std::vector<size_t> strides_ = {};
    };

} // namespace Breeze

#endif // CPU_TENSOR_H