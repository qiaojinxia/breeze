#include "CPUTensor.h"
#include <algorithm>
#include <stdexcept>

#include "CPUTensorOps.h"

namespace Breeze {
    template<typename T>
    CPUTensor<T>::~CPUTensor() {
        delete this->ops;
    }

    template<typename T>
    CPUTensor<T>::CPUTensor(const Shape& _shape)
        : Tensor<T>(_shape, Device::CPU,new CPUTensorOps<T>()),
          memory_block(std::make_shared<TensorStorage<T, CPUDevice>>(_shape.total_size())) {
    }

    template<typename T>
        CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data,
            const size_t offset, const std::vector<size_t>&& shape_size, const std::vector<size_t>&& strides)
            : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()), memory_block(data),offset_(offset), strides_(strides){
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator+(const Tensor<T>& rhs) const {
        return this->ops->add(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator-(const Tensor<T>& rhs) const {
        return this->ops->subtract(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator*(const Tensor<T>& rhs) const {
        return this->ops->multiply(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator/(const Tensor<T>& rhs) const {
        return this->ops->divide(*this, rhs);
    }

    template<typename T>
    [[nodiscard]] std::shared_ptr<Tensor<T>> CPUTensor<T>::matmul(const Tensor<T>& rhs) const {
        return this->ops->matmul(*this, rhs);
    }

    template<typename T>
    void CPUTensor<T>::broadcast(Tensor<T>& rhs) {
        this->ops->broadcastTensors(*this, rhs);
    }

    template<typename T>
    void CPUTensor<T>::resize(const Shape& new_shape) {
        if (new_shape.total_size() != this->shape.total_size()) {
            throw std::invalid_argument("New shape must have the same total size");
        }
        this->shape = new_shape;
        memory_block = std::make_shared<TensorStorage<T, CPUDevice>>(new_shape.total_size());
    }

    template<typename T>
    T* CPUTensor<T>::data() {
        return memory_block->getData();
    }

    template<typename T>
    const T* CPUTensor<T>::data() const {
        return memory_block->getData();
    }

    template<typename T>
    void CPUTensor<T>::to_cpu() {
        // Already on CPU, do nothing
    }

    template<typename T>
    void CPUTensor<T>::to_gpu() {
        throw std::runtime_error("GPU not supported for CPUTensor");
    }

    template<typename T>
    void CPUTensor<T>::fill(T value) {
        std::fill(data(), data() + this->size(), value);
    }

    template<typename T>
    void CPUTensor<T>::setTensorStorage(std::shared_ptr<TensorStorage<T, CPUDevice>> new_block,Shape&& n_shape) {
        memory_block = std::move(new_block);
        this->shape = std::move(n_shape); // 使用右值引用
    }

    template<typename T>
    void CPUTensor<T>::print(std::ostream& os) const {
        // 检查shape数组的有效性
        const auto& shape_ = this->shape.dims();
        if (shape_.empty() || std::any_of(shape_.begin(), shape_.end(), [](const size_t s) { return s == 0; })) {
            os << "Invalid shape dimensions." << std::endl;
            return;
        }

        std::function<void(const T*, const std::vector<size_t>&, size_t, const std::string&)> print_recursive;
        print_recursive = [&](const T* data_, const std::vector<size_t>& shape_, const size_t dim, const std::string& indent) {
            os << "[";
            if (dim == shape_.size() - 1) {
                for (size_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << ", ";
                    os << data_[i];
                }
                os << "]";
            } else {
                const auto dim_dt = static_cast<std::vector<size_t>::difference_type>(dim);
                const size_t stride = std::accumulate(shape_.begin() + dim_dt + 1, shape_.end(), 1ULL, std::multiplies<>());
                std::string new_indent = indent + " ";
                for (size_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << "\n" << new_indent;
                    print_recursive(data_ + i * stride, shape_, dim + 1, new_indent);
                    if (i < shape_[dim] - 1) os << ",";
                }
                os << "]";
            }
        };
        print_recursive(this->data(), shape_, 0, "");
        os << std::endl;
}

    template <typename T>
    [[nodiscard]] std::shared_ptr<Tensor<T>> CPUTensor<T>::slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const {
        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        int64_t new_offset = this->offset_;

        for (size_t i = 0; i < ranges.size(); ++i) {
            const std::pair<int64_t, int64_t> &slice = ranges[i];
            const int64_t dim_size = this->get_shape().dims()[i];
            int64_t start = slice.first;
            int64_t end = slice.second;
            int64_t step = 1; // 假设步长为1，如果需要可以添加步长参数

            // 处理负索引和默认值
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size;
            if (end > dim_size) end = dim_size;
            if (start >= end) {
                new_shape.push_back(0);
                new_strides.push_back(this->get_shape().strides()[i]);
                continue;
            }

            // 计算新的形状
            const size_t new_dim_size = (end - start + step - 1) / step;
            new_shape.push_back(new_dim_size);

            // 计算新的步长
            new_strides.push_back(this->get_shape().strides()[i] * step);

            // 更新偏移量
            new_offset += start * this->get_shape().strides()[i];
        }

        return std::make_shared<CPUTensor>(this->memory_block, new_offset,
            std::move(new_shape), std::move(new_strides));
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::view(const Shape& new_shape) const {
       return nullptr;
    }

// Explicit instantiation for common types
template class CPUTensor<float>;
template class CPUTensor<double>;

} // namespace Breeze