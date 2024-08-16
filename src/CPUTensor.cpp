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
        const std::vector<int32_t> steps(_shape.ndim(), 1);
        steps_ = steps;
    }

    template<typename T>
        CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data,
            const size_t offset, const std::vector<size_t>&& shape_size, std::vector<int32_t>&& steps)
            : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()),
            memory_block(data),offset_(offset),
            steps_(std::move(steps)){
        auto strides = Shape(shape_size).strides();
        this->strides_ = strides;
    }

    template<typename T>
        CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data,
            const size_t offset, const std::vector<size_t>&& shape_size, std::vector<int32_t>&& steps, std::vector<size_t>&& strides)
            : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()),
            memory_block(data),offset_(offset),
            steps_(std::move(steps)),strides_(std::move(strides)){
    }

    template<typename T>
      CPUTensor<T>::CPUTensor(std::shared_ptr<TensorStorage<T, CPUDevice>> data, const std::vector<size_t>&& shape_size)
          : Tensor<T>(Shape(shape_size), Device::CPU,new CPUTensorOps<T>()),
          memory_block(data){
        const std::vector<int32_t> steps(shape_size.size(),1);
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
    [[nodiscard]] std::vector<size_t> CPUTensor<T>::get_strides() const {
        if (!strides_.empty()) {
            return strides_;
        }
        return this->get_shape().strides();
    }

    template<typename T>
    [[nodiscard]] std::vector<int32_t> CPUTensor<T>::get_steps() const  {
        return this->steps_;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::reshape(const std::vector<int32_t>& new_shape) const {
        // 计算新形状的总元素数
        size_t new_size = 1;
        int32_t dynamic_dim = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                if (dynamic_dim != -1) {
                    throw std::invalid_argument("Only one dimension can be -1 in reshape");
                }
                dynamic_dim = static_cast<int32_t>(i);
            } else if (new_shape[i] < 0) {
                throw std::invalid_argument("Invalid negative dimension in reshape");
            } else {
                new_size *= new_shape[i];
            }
        }

        // 处理动态维度
        std::vector<int32_t> actual_new_shape = new_shape;
        if (dynamic_dim != -1) {
            if (this->size() % new_size != 0) {
                throw std::invalid_argument("New shape is not compatible with the number of elements");
            }
            actual_new_shape[dynamic_dim] = this->size() / new_size;
            new_size = this->size();
        }

        // 检查新旧大小是否匹配
        if (new_size != this->size()) {
            throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
        }

        // 如果张量是连续的，使用 view
        if (this->is_contiguous()) {
            return this->view(std::move(actual_new_shape));
        }

        // 如果张量不是连续的，先 clone 再 view
        auto cloned = this->clone();
        return cloned->view(std::move(actual_new_shape));

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
        fill([&](const std::vector<size_t>& _coords) {return value;});
    }

    template<typename T>
    void CPUTensor<T>::fill(const std::function<T(const std::vector<size_t>&)>& value_func) {
        std::vector<size_t> indices(this->get_shape().ndim(), 0);
        const auto& dims = this->get_shape().dims();
        const auto& strides = this->get_strides();
        const auto& steps = this->steps_;
        const size_t total_elements = this->get_shape().total_size();

        for (size_t i = 0; i < total_elements; ++i) {
            // 计算偏移量
            size_t offset = offset_;
            for (int32_t j = 0; j < indices.size(); ++j) {
                offset += indices[j] * strides[j] * steps_[j];
            }

            // 填充当前位置
            memory_block->getData()[offset] = value_func(indices);

            // 更新索引
            for (int32_t dim = static_cast<int32_t>(indices.size()) - 1; dim >= 0; --dim) {
                if (++indices[dim] < dims[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    template <typename T>
    bool CPUTensor<T>::is_contiguous() const {
            // 检查是否所有的步长都是1
            for (const auto& step : steps_) {
                if (step != 1) return false;
            }

            // 检查是否内存布局是连续的
            size_t expected_stride = 1;
            for (int i = strides_.size() - 1; i >= 0; --i) {
                if (strides_[i] == 0) {
                    return false;
                }
                if (strides_[i] != expected_stride) return false;
                expected_stride *= this->get_shape().dims()[i];
            }
            return true;
        }

    template<typename T>
    void CPUTensor<T>::setTensorStorage(std::shared_ptr<TensorStorage<T, CPUDevice>> new_block,Shape&& n_shape) {
        memory_block = std::move(new_block);
        const std::vector<int32_t> steps(n_shape.ndim(), 1);
        steps_ = steps;
        this->shape = std::move(n_shape);
    }

    template<typename T>
    const T& CPUTensor<T>::at(const std::vector<size_t>& indices) const{
        size_t offset = offset_;
        for (int32_t i = 0; i < indices.size(); ++i) {
            auto stride = this->get_strides()[i];
            offset += (indices[i] * stride) * steps_[i];
        }
        return memory_block->getData()[offset];
    }

    template<typename T>
    void CPUTensor<T>::set_value(const std::vector<size_t>& indices,T value){
        size_t offset = offset_;
        for (int32_t i = 0; i < indices.size(); ++i) {
            auto stride = this->get_strides()[i];
            offset += (indices[i] * stride) * steps_[i];
        }
        T* data_ptr = &memory_block->getData()[offset];
        *data_ptr = value;
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
    const std::vector<std::pair<int32_t, int32_t>>& ranges) const {
        std::vector<std::tuple<int32_t, int32_t, int32_t>> full_ranges;
        full_ranges.reserve(ranges.size());
        for (const auto& [start, end] : ranges) { // 使用结构化绑定
            full_ranges.emplace_back(start, end, 1);
        }
        return slice(full_ranges);
    }

    template <typename T>
    [[nodiscard]] std::shared_ptr<Tensor<T>> CPUTensor<T>::slice(
        const std::vector<std::tuple<int32_t, int32_t, int32_t>>& ranges) const {

        std::vector<size_t> new_shape;
        int32_t new_offset = this->offset_;
        std::vector<int32_t> new_steps;
        std::vector<size_t> new_strides;
        const auto& original_shape = this->get_shape().dims();
        const auto& original_strides = this->get_strides();

        for (size_t i = 0; i < ranges.size(); ++i) {
            const int32_t dim_size = original_shape[i];
            auto [start, end, step] = ranges[i];

            if (start == KEEP_ALL || (start == 0 && end == KEEP_ALL && step == 1)) {
                new_shape.push_back(dim_size);
                new_steps.push_back(step);
                new_strides.push_back(original_strides[i]);
                continue;
            }

            if (end == KEEP_REST) {
                end = dim_size;
            }

            if (start < 0) start += dim_size;
            if (end <= 0) end += dim_size;
            start = std::clamp(start, 0, dim_size);
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
                new_steps.push_back(step);
                new_offset += start * original_strides[i];
                new_strides.push_back(original_strides[i]);
            }

        }

        return std::make_shared<CPUTensor>(memory_block, new_offset,
            std::move(new_shape), std::move(new_steps), std::move(new_strides));
    }


    template <typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::clone() const {
            const auto& src_shape = this->shape.dims();
            const auto ndim = this->shape.ndim();
            const auto& src_strides = this->get_strides();
            const auto& src_steps = this->get_steps();
            const T* src_data = this->data();

            auto dst_tensor = std::make_shared<CPUTensor>(Shape{src_shape});
            T* dst_data = dst_tensor->data();

            // 快速路径：如果源张量是连续的，直接进行整体复制
            if (this->is_contiguous()) {
                std::memcpy(dst_data, src_data + this->offset_, this->size() * sizeof(T));
                return dst_tensor;
            }

            // 计算外部循环的维度
            size_t outer_dim = 1;
            for (size_t d = 0; d < ndim - 1; ++d) {
                outer_dim *= src_shape[d];
            }

            const size_t copy_size = src_shape.back();
            const size_t src_stride = src_strides[ndim-1] * src_steps[ndim-1];

    #pragma omp parallel for if(outer_dim > 1000)
            for (size_t i = 0; i < outer_dim; ++i) {
                size_t src_offset = offset_;
                size_t dst_offset = 0;
                size_t idx = i;

                for (int32_t j = ndim - 2; j >= 0; --j) {
                    size_t _idx_ = idx % src_shape[j];
                    idx /= src_shape[j];
                    src_offset += _idx_ * src_strides[j] * src_steps[j];
                    dst_offset += _idx_ * dst_tensor->get_strides()[j];
                }

                if constexpr (std::is_same_v<T, float>) {
                    cblas_scopy(copy_size, src_data + src_offset, src_stride, dst_data + dst_offset, 1);
                } else if constexpr (std::is_same_v<T, double>) {
                    cblas_dcopy(copy_size, src_data + src_offset, src_stride, dst_data + dst_offset, 1);
                } else {
                    std::copy_n(src_data + src_offset, copy_size, dst_data + dst_offset);
                }
            }

            return dst_tensor;
        }

    template <typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::view(const std::vector<int32_t>& new_shape) const {
            // 检查张量是否连续
            if (!is_contiguous()) {
                throw std::runtime_error("Cannot perform view on non-contiguous tensor");
            }

            // 处理自动计算维度
            const int32_t total_size = this->get_shape().total_size();
            int32_t new_total_size = 1;
            int32_t dynamic_dim = -1;

            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (new_shape[i] == -1) {
                    if (dynamic_dim != -1) {
                        throw std::invalid_argument("Only one dimension can be -1 in view");
                    }
                    dynamic_dim = static_cast<int32_t>(i);
                } else if (new_shape[i] < 0) {
                    throw std::invalid_argument("Invalid negative dimension in view");
                } else {
                    new_total_size *= new_shape[i];
                }
            }

            // 计算动态维度
            std::vector<size_t> final_shape;
            final_shape.reserve(new_shape.size());

            if (dynamic_dim != -1) {
                if (total_size % new_total_size != 0) {
                    throw std::invalid_argument("New shape is not compatible with the number of elements");
                }
                for (size_t i = 0; i < new_shape.size(); ++i) {
                    if (i == dynamic_dim) {
                        final_shape.push_back(total_size / new_total_size);
                    } else {
                        final_shape.push_back(new_shape[i]);
                    }
                }
            } else {
                // 检查新形状是否合法
                if (new_total_size != total_size) {
                    throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
                }
                for (const auto& dim : new_shape) {
                    final_shape.push_back(dim);
                }
            }

            // 返回新的张量视图
            return std::make_shared<CPUTensor>(memory_block, std::move(final_shape));
        }


    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::unsqueeze(const int32_t dim) const {
        const auto& original_shape = this->get_shape().dims();
        const auto& original_strides = this->get_strides();
        const auto& original_steps = this->get_steps();

        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        std::vector<int32_t> new_steps;

        new_shape.reserve(original_shape.size() + 1);
        new_strides.reserve(original_shape.size() + 1);
        new_steps.reserve(original_shape.size() + 1);

        // 处理负数维度
        const int32_t adjusted_dim = dim < 0 ? static_cast<int32_t>(original_shape.size()) + dim + 1 : dim;

        for (int32_t i = 0; i < original_shape.size() + 1; ++i) {
            if (i == adjusted_dim) {
                new_shape.push_back(1);
                new_strides.push_back(1);
                new_steps.push_back(1);
            }

            if (i < original_shape.size()) {
                new_shape.push_back(original_shape[i]);
                new_strides.push_back(original_strides[i]);
                new_steps.push_back(original_steps[i]);
            }
        }


        new_strides = Utils::compute_strides_with_zeros(new_shape, new_strides);

        return std::make_shared<CPUTensor<T>>(this->memory_block, this->offset_,
           std::move(new_shape), std::move(new_steps), std::move(new_strides));
    }

    template<typename T>
 std::shared_ptr<Tensor<T>> CPUTensor<T>::squeeze(const int32_t dim) const {
        const auto& original_shape = this->get_shape().dims();
        const auto& original_strides = this->get_strides();
        const auto& original_steps = this->get_steps();

        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        std::vector<int32_t> new_steps;

        new_shape.reserve(original_shape.size());
        new_strides.reserve(original_shape.size());
        new_steps.reserve(original_shape.size());

        const int32_t adjusted_dim = dim < 0 ? static_cast<int32_t>(original_shape.size()) + dim : dim;

        for (size_t i = 0; i < original_shape.size(); ++i) {
            // 如果 不是当前维度 维度不是1 或者广播 维度 都去不掉
            if (static_cast<int32_t>(i) != adjusted_dim || original_shape[i] != 1 || original_strides[i] == 0) {
                new_shape.push_back(original_shape[i]);
                new_strides.push_back(original_strides[i]);
                new_steps.push_back(original_steps[i]);
            }
        }

        // 如果所有维度都被移除，保留一个维度
        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
            new_steps.push_back(1);
        }

        // 使用新函数重新计算strides
        new_strides = Utils::compute_strides_with_zeros(new_shape, new_strides);

        return std::make_shared<CPUTensor>(this->memory_block, this->offset_,
            std::move(new_shape), std::move(new_steps), std::move(new_strides));
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
        std::vector<int32_t> new_steps(new_dims.size(), 1);

        // 从右到左填充新的 strides 和 steps
        int32_t current_dim = current_shape.size() - 1;
        for (int32_t i = static_cast<int32_t>(new_dims.size()) - 1; i >= 0; --i) {
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

    template <typename T>
    std::shared_ptr<CPUTensor<T>> CPUTensor<T>::arrange(T  begin,T end,T step ) {
        const auto m_size = static_cast<size_t>((end - begin + 1) /  step);
        auto tensor = std::make_shared<CPUTensor>(Shape{std::vector{m_size}});
        for(size_t i = 0; i< tensor->get_shape().total_size(); ++i) {
            auto data_ptr = tensor->data() + i;
            *data_ptr = begin + i * step;
        }
        return tensor;
    }

    template <typename T>
    std::shared_ptr<CPUTensor<T>> CPUTensor<T>::cat(const std::vector<CPUTensor*>& tensors, int32_t dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("No tensors provided for concatenation");
        }

        int32_t ndim = tensors[0]->get_shape().ndim();
        for (const auto& tensor : tensors) {
            if (tensor->get_shape().ndim() != ndim) {
                throw std::invalid_argument("All tensors must have the same number of dimensions");
            }
        }

        if (dim < 0) {
            dim += ndim;
        }

        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("Invalid dimension for concatenation");
        }

        std::vector<size_t> new_shape = tensors[0]->get_shape().dims();
        size_t concat_dim_size = 0;
        for (const auto& tensor : tensors) {
            concat_dim_size += tensor->get_shape().dims()[dim];
            for (size_t i = 0; i < ndim; ++i) {
                if (i != dim && tensor->get_shape().dims()[i] != new_shape[i]) {
                    throw std::invalid_argument("Inconsistent tensor shapes");
                }
            }
        }
        new_shape[dim] = concat_dim_size;

        auto result = std::make_shared<CPUTensor>(Shape{new_shape});
        T* result_data = result->data();

        // 计算每次复制的块大小（不包括最后一维）
        size_t block_size = 1;
        for (int i = dim + 1; i < ndim - 1; ++i) {
            block_size *= new_shape[i];
        }

        // 最后一维的大小
        const size_t last_dim_size = new_shape[ndim - 1];

        // 计算外层循环的次数和步长
        std::vector<size_t> outer_steps(dim);
        size_t outer_iterations = 1;
        for (int i = 0; i < dim; ++i) {
            outer_steps[i] = outer_iterations;
            outer_iterations *= new_shape[i];
        }

        size_t result_offset = 0;
        for (size_t i = 0; i < outer_iterations; ++i) {
            for (const auto& tensor : tensors) {
                const T* src_data = tensor->data();
                const std::vector<size_t>& src_shape = tensor->get_shape().dims();
                const std::vector<size_t>& src_strides = tensor->get_strides();
                const std::vector<int32_t>& src_steps = tensor->get_steps();

                const size_t src_dim_size = src_shape[dim];

                // 计算源张量的起始偏移，考虑所有外层维度
                size_t src_offset = tensor->offset_;
                for (int d = 0; d < dim; ++d) {
                    const size_t idx = (i / outer_steps[d]) % new_shape[d];
                    src_offset += idx * src_strides[d] * src_steps[d];
                }

                if (tensor->is_contiguous() && result->is_contiguous()) {
                    // 使用 BLAS 进行连续内存的复制
                    size_t copy_size = src_dim_size * block_size * last_dim_size;
                    if constexpr (std::is_same_v<T, float>) {
                        cblas_scopy(copy_size, src_data + src_offset, 1, result_data + result_offset, 1);
                    } else if constexpr (std::is_same_v<T, double>) {
                        cblas_dcopy(copy_size, src_data + src_offset, 1, result_data + result_offset, 1);
                    } else {
                        std::memcpy(result_data + result_offset, src_data + src_offset, copy_size * sizeof(T));
                    }
                    result_offset += copy_size;
                } else {
                    // 对于非连续存储，我们使用 BLAS 复制最后一维
                    for (size_t j = 0; j < src_dim_size * block_size; ++j) {
                        size_t current_src_offset = src_offset,dst_offset = result_offset;
                        auto idx = j;
                        for (int d = ndim - 2; d >= dim; --d) {
                            size_t coord = idx % src_shape[d];
                            idx /= src_shape[d];
                            current_src_offset += coord * src_strides[d] * src_steps[d];
                            dst_offset += coord * result->get_strides()[d];
                        }

                        if constexpr (std::is_same_v<T, float>) {
                            cblas_scopy(last_dim_size, src_data + current_src_offset, src_strides[ndim-1] * src_steps[ndim-1],
                                        result_data + dst_offset, 1);
                        } else if constexpr (std::is_same_v<T, double>) {
                            cblas_dcopy(last_dim_size, src_data + current_src_offset, src_strides[ndim-1] * src_steps[ndim-1],
                                        result_data + dst_offset, 1);
                        } else {
                            for (size_t k = 0; k < last_dim_size; ++k) {
                                result_data[dst_offset + k] = src_data[current_src_offset + k * src_strides[ndim-1] * src_steps[ndim-1]];
                            }
                        }
                    }
                    result_offset += src_dim_size * block_size * last_dim_size;
                }
            }
        }

        return result;
    }
// Explicit instantiation for common types
template class CPUTensor<float>;
template class CPUTensor<double>;

} // namespace Breeze