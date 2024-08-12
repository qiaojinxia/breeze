#include "CPUTensor.h"
#include <algorithm>
#include <stdexcept>

#include "CPUTensorOps.h"
#include "common/Const.h"
namespace Breeze {
    template<typename T>
    CPUTensor<T>::~CPUTensor() {
        delete this->ops;
    }

    template<typename T>
    CPUTensor<T>::CPUTensor(const Shape& _shape)
    : Tensor<T>(_shape, Device::CPU, new CPUTensorOps<T>()),
      memory_block(std::make_shared<TensorStorage<T, CPUDevice>>(_shape.total_size())) {
        const std::vector<int64_t> steps(_shape.ndim(), 1);
        steps_ = steps;
    }

    template<typename T>
        CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data,
            const size_t offset, const std::vector<size_t>&& shape_size, std::vector<int64_t> steps, const bool contiguous)
            : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()),
            memory_block(data),offset_(offset),
            strides_(std::accumulate(shape_size.begin(), shape_size.end(), 1ULL, std::multiplies<>())),
            steps_(std::move(steps)), is_contiguous_(contiguous){
    }

    template<typename T>
      CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, const std::vector<size_t>&& shape_size)
          : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()),
          memory_block(data){
        const std::vector<int64_t> steps(shape_size.size(),1);
        steps_ = steps;
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
    [[nodiscard]] std::vector<size_t> CPUTensor<T>::get_strides() const {
        if (!strides_.empty()) {
            return strides_;
        }
        return this->get_shape().strides();
    }

    template<typename T>
    [[nodiscard]] std::vector<int64_t> CPUTensor<T>::get_steps() const  {
        return this->steps_;
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
        const std::vector<int64_t> steps(n_shape.ndim(), 1);
        steps_ = steps;
        this->shape = std::move(n_shape);
    }

    template<typename T>
    const T& CPUTensor<T>::at(const std::vector<size_t>& indices) const {
        size_t offset = offset_;
        for (int64_t i = 0; i < indices.size(); ++i) {
            auto stride = this->get_strides()[i];
            offset += (indices[i] * stride) * steps_[i];
        }
        return memory_block->getData()[offset];
    }

    template<typename T>
    void CPUTensor<T>::print(std::ostream& os) const {
        const auto& shape_ = this->shape.dims();
        if (shape_.empty()) {
            os << "Empty tensor" << std::endl;
            return;
        }

        std::function<void(const std::vector<size_t>&, size_t, const std::string&)> print_recursive;
        print_recursive = [&](const std::vector<size_t>& indices, size_t dim, const std::string& indent) {
            os << "[";
            if (dim == shape_.size() - 1) {
                for (size_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << ", ";
                    std::vector<size_t> current_indices = indices;
                    current_indices.push_back(i);
                    os << this->at(current_indices);
                }
                os << "]";
            } else {
                const std::string new_indent = indent + " ";
                for (size_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << "\n" << new_indent;
                    std::vector<size_t> current_indices = indices;
                    current_indices.push_back(i);
                    print_recursive(current_indices, dim + 1, new_indent);
                    if (i < shape_[dim] - 1) os << ",";
                }
                os << "]";
            }
        };

        print_recursive({}, 0, "");
        os << std::endl;
    }

    template <typename T>
    [[nodiscard]] std::shared_ptr<Tensor<T>> CPUTensor<T>::slice(
    const std::vector<std::pair<int64_t, int64_t>>& ranges) const {
        std::vector<std::tuple<int64_t, int64_t, int64_t>> full_ranges;
        full_ranges.reserve(ranges.size());
        for (const auto& [start, end] : ranges) { // 使用结构化绑定
            full_ranges.emplace_back(start, end, 1);
        }
        return slice(full_ranges);
    }

    template <typename T>
    [[nodiscard]] std::shared_ptr<Tensor<T>> CPUTensor<T>::slice(
        const std::vector<std::tuple<int64_t, int64_t, int64_t>>& ranges) const {

        std::vector<size_t> new_shape;
        int64_t new_offset = this->offset_;
        std::vector<int64_t> new_steps;

        const auto& original_shape = this->get_shape().dims();
        const auto& original_strides = this->get_shape().strides();
        bool new_is_contiguous = true;

        for (size_t i = 0; i < ranges.size(); ++i) {
            const int64_t dim_size = original_shape[i];
            auto [start, end, step] = ranges[i];

            if (step != 1) new_is_contiguous = false;

            if (start == KEEP_ALL || (start == 0 && end == KEEP_ALL && step == 1)) {
                new_shape.push_back(dim_size);
                new_steps.push_back(step);
                continue;
            }

            if (end == KEEP_REST) {
                end = dim_size;
            }

            if (start < 0) start += dim_size;
            if (end <= 0) end += dim_size;
            start = std::clamp(start, static_cast<int64_t>(0), dim_size);
            end = std::clamp(end, start, dim_size);

            if (step == 0) {
                throw std::invalid_argument("Slice step cannot be zero");
            }

            if (step < 0) {
                if (start < end) {
                    std::swap(start, end);
                }
                start = dim_size - 1 - start;
                end = dim_size - 1 - end;
                step = -step;
            }

            if (const size_t new_dim_size = (end - start + step - 1) / step; new_dim_size > 0) {
                new_shape.push_back(new_dim_size);
                new_steps.push_back(steps_[i] * step);
                new_offset += start * original_strides[i];
            }
        }

        return std::make_shared<CPUTensor>(this->memory_block, new_offset,
            std::move(new_shape), std::move(new_steps), new_is_contiguous);
    }

    template <typename T>
    bool CPUTensor<T>::is_contiguous() const {
           return is_contiguous_;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::view(std::vector<size_t>&& new_shape) const {
        // 检查张量是否连续
        if (!this->is_contiguous()) {
            throw std::logic_error("View operation is only valid on contiguous tensors.");
        }

        // 检查新形状是否合法
        size_t new_total_size = 1;
        for (const auto& dim : new_shape) {
            new_total_size *= dim;
        }

        if (new_total_size != this->get_shape().total_size()) {
            throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
        }

        // 返回新的张量视图
        return std::make_shared<CPUTensor>(this->memory_block, std::move(new_shape));
    }


    template <typename T>
    void CPUTensor<T>::expand(const Shape&& new_shape)  {
        const auto& current_shape = this->shape.dims();
        const auto& new_dims = new_shape.dims();

        // 检查新形状是否兼容
        if (new_dims.size() < current_shape.size()) {
            throw std::invalid_argument("New shape must have at least as many dimensions as the current shape");
        }

        std::vector<size_t> new_strides(new_dims.size(), 0);
        std::vector<int64_t> new_steps(new_dims.size(), 1);

        // 从右到左填充新的 strides 和 steps
        int64_t current_dim = current_shape.size() - 1;
        for (int64_t i = static_cast<int64_t>(new_dims.size()) - 1; i >= 0; --i) {
            if (current_dim >= 0) {
                if (new_dims[i] == current_shape[current_dim]) {
                    new_strides[i] = this->get_strides()[current_dim];
                    --current_dim;
                } else if (current_shape[current_dim] == 1) {
                    new_strides[i] = 0;  // 广播维度的 stride 为 0
                    --current_dim;
                } else {
                    throw std::invalid_argument("Incompatible shapes for expansion");
                }
            } else {
                new_strides[i] = 0;  // 新增维度的 stride 为 0
            }
        }
        if (current_dim >= 0) {
            throw std::invalid_argument("Incompatible shapes for expansion");
        }
        this->shape = new_shape;
        this->strides_ = new_strides;
    }

// Explicit instantiation for common types
template class CPUTensor<float>;
template class CPUTensor<double>;

} // namespace Breeze