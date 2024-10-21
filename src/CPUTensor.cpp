#include "CPUTensor.h"
#include <stdexcept>
#include "omp.h"
#include <cblas.h>
#include "CPUTensorOps.h"
#include <iomanip>

namespace Breeze {
    template<typename ScalarType>
    CPUTensor<ScalarType>::~CPUTensor() = default;

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(Shape shape)
        : Tensor<ScalarType>(std::move(shape), Device::CPU, TensorState::ALLOCATED),
          memory_block_(std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size())) {
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(std::vector<index_t> shape)
        : Tensor<ScalarType>(Shape(std::move(shape)), Device::CPU, TensorState::ALLOCATED) {
        memory_block_ = std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size());
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(const std::initializer_list<index_t> shape)
        : Tensor<ScalarType>(Shape(shape), Device::CPU, TensorState::ALLOCATED),
          memory_block_(std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size())) {
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(const std::vector<ScalarType>& elements ,const std::initializer_list<index_t> shape)
        : Tensor<ScalarType>(Shape(shape), Device::CPU, TensorState::ALLOCATED),
          memory_block_(std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size())) {
        this->strides_ = this->get_shape().compute_strides();
        ScalarType* dst_offset = this->mutable_data();
        for (auto element : elements)
        {
            *(dst_offset++) = element;
        }
    }


    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(const CPUTensor& other)
        : Tensor<ScalarType>(Shape(std::vector<index_t>(other.get_shape().dims().begin(), other.get_shape().dims().end())), Device::CPU, TensorState::ALLOCATED),
          memory_block_(other.memory_block_) {
        this->offset_ = other.offset_;  // Use base class offset_
        this->strides_ = other.strides_;  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(const CPUTensor& other, std::vector<index_t>&& shape)
        : Tensor<ScalarType>(Shape(shape), Device::CPU, TensorState::ALLOCATED),
          memory_block_(other.memory_block_) {
        this->offset_ = other.offset_;  // Use base class offset_
        this->strides_ = std::move(other.strides_);  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(Shape shape, ScalarType value)
        : Tensor<ScalarType>(std::move(shape), Device::CPU, TensorState::INITIALIZED),
          memory_block_(std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size())) {
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
        fill(value);
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor()
        : Tensor<ScalarType>(Shape{}, Device::CPU, TensorState::UNINITIALIZED),
          memory_block_() {}

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data,
                            const index_t offset, std::vector<index_t> shape, std::vector<index_t> strides)
        : Tensor<ScalarType>(Shape(std::move(shape)), Device::CPU, TensorState::ALLOCATED),
          memory_block_(data) {
        this->offset_ = offset;  // Use base class offset_
        this->strides_ = std::move(strides);  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data,
                            const index_t offset, std::vector<index_t> shape)
        : Tensor<ScalarType>(Shape(std::move(shape)), Device::CPU, TensorState::ALLOCATED),
          memory_block_(data) {
        this->offset_ = offset;  // Use base class offset_
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
    }

    template<typename ScalarType>
    CPUTensor<ScalarType>::CPUTensor(std::shared_ptr<TensorStorage<ScalarType, CPUDevice>> data, std::vector<index_t> shape)
        : Tensor<ScalarType>(Shape(std::move(shape)), Device::CPU,TensorState::ALLOCATED),
          memory_block_(data) {
        this->strides_ = this->get_shape().compute_strides();  // Use base class strides_
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::set_initial_shape(Shape& shape) {
        this->shape = std::move(shape);
        memory_block_ = std::make_shared<TensorStorage<ScalarType, CPUDevice>>(this->get_shape().total_size());
        this->set_state(TensorState::ALLOCATED);
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::set_shape_and_strides(std::vector<index_t>& shape,
        std::vector<index_t>& strides,const bool reduce) {
        // 计算新形状的总元素数量
        // 获取当前张量的总元素数量
        index_t current_total_size = this->shape.total_size();
        if (!reduce) {
            index_t new_total_size = std::accumulate(shape.begin(), shape.end(),
                                                            static_cast<index_t>(1), std::multiplies<>());
            if (shape.empty()) {
                new_total_size = 1;
            }
            // 使用 BREEZE_ASSERT 检查新形状的总元素数量是否与当前张量的总元素数量相同
            BREEZE_ASSERT(new_total_size == current_total_size,
                          "View shape mismatch: new shape total size (", new_total_size,
                          ") must be equal to current tensor total size (", current_total_size, ")");
        }
        if (!strides.empty())
            this->strides_ = strides;
        // 如果断言通过，更新形状
        this->shape = Shape(std::move(shape));
    }

    DEFINE_UNARY_OP(sin)
    DEFINE_UNARY_OP(cos)
    DEFINE_UNARY_OP(tan)
    DEFINE_UNARY_OP(atan)
    DEFINE_UNARY_OP(log)
    DEFINE_UNARY_OP(log2)
    DEFINE_UNARY_OP(log10)
    DEFINE_UNARY_OP(exp)
    DEFINE_UNARY_OP(sqrt)
    DEFINE_UNARY_OP(rsqrt)
    DEFINE_UNARY_OP(abs)

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::sum(std::vector<index_t> dims, const bool keep_dim) {
        const index_t ndim = this->shape.ndim();
        // 处理标量sum
        if (ndim == 0) {
            auto result = std::make_shared<CPUTensor>(Shape{});
            result->mutable_data()[0] = this->data()[0];
            return result;
        }
        // 如果维度为空，则返回整个张量的和
        if (dims.empty()) {
            std::vector<index_t> all_dim = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().sum(*this, all_dim, keep_dim);
        }
        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
                "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }
        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the sum operation.");
        // 执行求和操作
        return CPUTensorOps<ScalarType>::getInstance().sum(*this, dims, keep_dim);
    }

    template<typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::max(std::vector<index_t> dims, const bool keep_dim) {
        const index_t ndim = this->shape.ndim();
        // 处理标量max
        if (ndim == 0) {
            auto result = std::make_shared<CPUTensor>(Shape{});
            result->mutable_data()[0] = this->data()[0];
            return result;
        }

        // 如果维度为空，则返回整个张量的最大值
        if (dims.empty()) {
            std::vector<index_t> all_dim = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().max(*this, all_dim, keep_dim);
        }

        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
                "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }

        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the max operation.");

        // 执行最大值操作
        return CPUTensorOps<ScalarType>::getInstance().max(*this, dims, keep_dim);
    }

    template<typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::min(std::vector<index_t> dims, const bool keep_dim) {
        const index_t ndim = this->shape.ndim();
        // 处理标量min
        if (ndim == 0) {
            auto result = std::make_shared<CPUTensor>(Shape{});
            result->mutable_data()[0] = this->data()[0];
            return result;
        }

        // 如果维度为空，则返回整个张量的最小值
        if (dims.empty()) {
            std::vector<index_t> all_dim = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().min(*this, all_dim, keep_dim);
        }

        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
                "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }

        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the min operation.");

        // 执行最小值操作
        return CPUTensorOps<ScalarType>::getInstance().min(*this, dims, keep_dim);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::std(std::vector<index_t> dims, bool keep_dim, bool unbiased) {
        const index_t ndim = this->shape.ndim();

        BREEZE_ASSERT(ndim > 0, "Cannot perform std on a scalar tensor as it contains only one element with no spread of values.");

        // 如果维度为空，则返回整个张量的标准差
        if (dims.empty()) {
            std::vector<index_t> all_dim = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().std(*this, all_dim, keep_dim, unbiased);
        }

        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
                "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }

        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the std operation.");

        // 执行标准差操作
        return CPUTensorOps<ScalarType>::getInstance().std(*this, dims, keep_dim, unbiased);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::var(std::vector<index_t> dims, bool keep_dim, bool unbiased) {
        const index_t ndim = this->shape.ndim();

        BREEZE_ASSERT(ndim > 0, "Cannot perform var on a scalar tensor as it contains only one element, which by definition has a variance of zero.");

            // 如果维度为空，则返回整个张量的方差
        if (dims.empty()) {
            std::vector<index_t> all_dims = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().var(*this, all_dims, keep_dim, unbiased);
        }

        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
            "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }
        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the var operation.");

        // 执行方差操作
        return CPUTensorOps<ScalarType>::getInstance().var(*this, dims, keep_dim, unbiased);
    }


    template<typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::mean(std::vector<index_t> dims, const bool keep_dim) {
        const index_t ndim = this->shape.ndim();

        if (ndim == 0) {
            auto result = std::make_shared<CPUTensor>(Shape{});
            result->mutable_data()[0] = this->data()[0];
            return result;
        }
        // 如果维度为空，则返回整个张量的平均值
        if (dims.empty()) {
            std::vector<index_t> all_dim = Utils::create_index_sequence(ndim);
            return CPUTensorOps<ScalarType>::getInstance().mean(*this, all_dim, keep_dim);
        }

        // 检查维度的有效性
        for (const auto& dim : dims) {
            BREEZE_ASSERT(dim >= 0 && dim < ndim,
                "Dimension index " + std::to_string(dim) + " is out of bounds for tensor with " + std::to_string(ndim) + " dimensions.");
        }

        // 检查维度是否重复
        std::vector<index_t> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        BREEZE_ASSERT(std::unique(sorted_dims.begin(), sorted_dims.end()) == sorted_dims.end(),
            "Duplicate dimensions are not allowed in the mean operation.");

        // 执行平均值操作
        return CPUTensorOps<ScalarType>::getInstance().mean(*this, dims, keep_dim);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::sum(std::vector<index_t> dims) {
        return sum(std::move(dims),false);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::max(std::vector<index_t> dims) {
        return max(std::move(dims),false);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::mean(std::vector<index_t> dims) {
        return mean(std::move(dims),false);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::min(std::vector<index_t> dims) {
        return min(std::move(dims),false);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::std(std::vector<index_t> dims) {
        return std(std::move(dims), false, true);
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::var(std::vector<index_t> dims) {
        return var(std::move(dims), false, true);
    }

    DEFINE_BINARY_OP(pow, pow)
    DEFINE_BINARY_OP(matmul, matmul)
    DEFINE_BINARY_OP(operator+, add)
    DEFINE_BINARY_OP(operator-, subtract)
    DEFINE_BINARY_OP(operator*, multiply)
    DEFINE_BINARY_OP(operator/, divide)
    DEFINE_BINARY_OP_VOID(operator+=, add_inplace)
    DEFINE_BINARY_OP_VOID(operator-=, subtract_inplace)
    DEFINE_BINARY_OP_VOID(operator*=, multiply_inplace)
    DEFINE_BINARY_OP_VOID(operator/=, divide_inplace)


    template<typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::randn(std::vector<index_t> shape) {
        // 创建一个新的 CPUTensor
        auto tensor = std::make_shared<CPUTensor>(Shape(std::move(shape)));
        CPUTensorOps<ScalarType>::getInstance().randn(*tensor);
        return tensor;
    }

    template<typename ScalarType>
    ScalarType CPUTensor<ScalarType>::operator[](const std::string& index) const{
        std::cout << index << std::endl;
        return this->at({1,2,3});
    }

    template<typename ScalarType>
    index_t CPUTensor<ScalarType>::n_bytes() const  {
        return memory_block_->total_bytes();
    }

    template<typename ScalarType>
    [[nodiscard]] std::vector<index_t> CPUTensor<ScalarType>::get_strides() const {
        if (!this->strides_.empty()) {
            return this->strides_;
        }
        return this->get_shape().strides();
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::reshape(const std::vector<index_t>& new_shape) const {
        if (new_shape.empty()) {
            throw std::invalid_argument("New shape cannot be empty");
        }

        const auto new_shape_ndim = static_cast<index_t>(new_shape.size());

        index_t computed_size = 1;
        index_t inferred_dim_index = -1;
        std::vector<index_t> final_shape;
        final_shape.reserve(new_shape_ndim);
        const index_t original_total_elements = this->get_shape().total_size();

        for (index_t i = 0; i < new_shape_ndim; ++i) {
            if (new_shape[i] == -1) {
                if (inferred_dim_index != -1) {
                    throw std::invalid_argument("Only one dimension can be inferred (-1) in reshape_view");
                }
                inferred_dim_index = i;
                final_shape.push_back(-1);
            } else if (new_shape[i] > 0) {
                final_shape.push_back(new_shape[i]);
                computed_size *= new_shape[i];
            } else {
                throw std::invalid_argument(Utils::Format("Invalid shape dimension %d", new_shape[i]));
            }
        }

        // 处理需要推断的维度
        if (inferred_dim_index != -1) {
            if (original_total_elements % computed_size != 0) {
                throw std::invalid_argument(Utils::Format("Cannot reshape tensor of %d elements into the requested shape", original_total_elements));
            }
            final_shape[inferred_dim_index] = original_total_elements / computed_size;
            computed_size *= final_shape[inferred_dim_index];
        }

        // 验证新旧元素总数是否匹配
        if (computed_size != original_total_elements) {
            throw std::invalid_argument(Utils::Format("Cannot reshape tensor of %d elements into the requested shape", original_total_elements));
        }

        // 处理标量的特殊情况
        if (original_total_elements == 1) {
            return std::make_shared<CPUTensor>(memory_block_, this->offset_, final_shape);
        }

        // 如果张量是连续的，直接创建新视图
        if (this->is_contiguous()) {
            return this->view(final_shape);
        }

        // 如果张量不是连续的，先创建副本再创建视图
        return this->clone()->view(std::move(final_shape));
    }


    template<typename ScalarType>
    ScalarType* CPUTensor<ScalarType>::mutable_data() {
        return memory_block_->data();
    }

    template<typename ScalarType>
    [[nodiscard]] const ScalarType* CPUTensor<ScalarType>::data() const {
        if (memory_block_ == nullptr)
            return nullptr;
        return memory_block_->data();
    }

    template<typename ScalarType>
    [[nodiscard]] index_t CPUTensor<ScalarType>::align_size() const {
       return memory_block_->total_size();
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::to_cpu() {
        // Already on CPU, do nothing
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::to_gpu() {
        throw std::runtime_error("GPU not supported for CPUTensor");
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::fill(ScalarType value) {
        CPUTensorOps<ScalarType>::getInstance().fill(*this, value);
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::fill(const std::function<ScalarType(const std::vector<index_t>&)>& value_func) {
        std::vector<index_t> indices(this->get_shape().ndim(), 0);
        const std::vector<index_t>& shape = this->get_shape().dims();
        const std::vector<index_t>& strides = this->get_strides();
        const index_t total_elements = this->get_shape().total_size();

        for (index_t i = 0; i < total_elements; ++i) {
            // 计算偏移量
            index_t offset = this->offset_;
            for (size_t j = 0; j < indices.size(); ++j) {
                offset += indices[j] * strides[j];
            }
            // 填充当前位置
            memory_block_->data()[offset] = value_func(indices);
            // 更新索引
            for (index_t dim = static_cast<index_t>(indices.size()) - 1; dim >= 0; --dim) {
                if (++indices[dim] < shape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    template <typename ScalarType>
    bool CPUTensor<ScalarType>::is_contiguous() const {
       return this->is_contiguous_in_range(0, -1);
    }

    template<typename ScalarType>
    const ScalarType& CPUTensor<ScalarType>::at(const std::vector<index_t>& indices) const {
        index_t offset = this->offset_;
        auto strides = this->get_strides();
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides[i];
        }
        return memory_block_->data()[offset];
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::set_value(const std::vector<index_t>& indices, ScalarType value) {
        index_t offset = this->offset_;
        const std::vector<index_t>& strides = this->get_strides();
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides[i];
        }
        ScalarType* data_ptr = &memory_block_->data()[offset];
        *data_ptr = value;
    }

    template<typename ScalarType>
    void CPUTensor<ScalarType>::print(std::ostream& os) const {
        std::cout << std::fixed << std::setprecision(6);
        const auto& shape_ = this->shape.dims();
        os << "tensor(";
        if (this->shape.ndim() == 0) {
            os << this->data()[0];
            os << ")" << std::endl;
            return;
        }
        std::function<void(const std::vector<index_t>&, size_t, const std::string&)> print_recursive;
        print_recursive = [&](const std::vector<index_t>& indices, index_t dim, const std::string& indent) {
            os << "[";
            if (dim == static_cast<index_t>(shape_.size()) - 1) {
                for (index_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << ", ";
                    std::vector<index_t> current_indices = indices;
                    current_indices.push_back(i);
                    os << this->at(current_indices);
                }
                os << "]";
            } else {
                const std::string new_indent = indent + " ";
                for (index_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) os << "\n" << new_indent;
                    std::vector<index_t> current_indices = indices;
                    current_indices.push_back(i);
                    print_recursive(current_indices, dim + 1, new_indent);
                    if (i < shape_[dim] - 1) os << ",";
                }
                os << "]";
            }
        };
        print_recursive({}, 0, "");
        os << ")" << std::endl;
    }

    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::slice(const std::vector<std::string>& range_strings) {

        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& original_strides = this->get_strides();
        const index_t original_ndim = this->get_shape().ndim();

        if (original_shape.empty()) {
            return this->shared_from_this();
        }

        if (range_strings.size() > original_shape.size()) {
            throw std::invalid_argument(Utils::Format("too many indices for tensor of dimension %d", original_shape.size()));
        }

        std::vector<index_t> final_shape;
        index_t new_offset = this->offset_;
        std::vector<index_t> new_strides;
        new_strides.reserve(original_ndim);

        for (index_t i = 0; i < static_cast<index_t>(range_strings.size()); ++i) {
            const index_t dim_size = original_shape[i];
            auto slice_params = Utils::parseSliceString(range_strings[i], dim_size);

            index_t start = slice_params[0];
            index_t end = slice_params[1];
            const index_t step = slice_params[2];
            if (step == 0) {
                throw std::invalid_argument("Slice step cannot be zero");
            }

            if (start < 0) {
                start = start + dim_size >= 0 ? start + dim_size : 0;
            } else {
                start = std::min(start, dim_size - 1);
            }

            if (end < 0) {
                if (step < 0) {
                    end = end + dim_size + 2 > 0  ? (end + dim_size + 1) % (dim_size + 1) : -1;
                    end == dim_size ? end = -1 : 0;
                }else {
                    end += dim_size;
                }
            } else {
                end = std::min(end, dim_size);
            }

            index_t new_dim_size;
            if (step > 0) {
                if (start >= end) {
                    new_dim_size = 0;
                } else {
                    new_dim_size = (end - start -1) / step + 1;
                }
            } else {
                if (start <= end) {
                    // 空切片
                    new_dim_size = 0;
                } else {
                    new_dim_size = ((start - end -1) / -step) + 1;
                }
            }
            new_offset += start * original_strides[i];
            final_shape.push_back(new_dim_size);
            auto stride  =  new_dim_size == 1 ? original_strides[i]: original_strides[i] * step;
            new_strides.push_back(stride);
        }

        // 处理未指定的维度（保持原样）
        for (auto i = static_cast<index_t>(range_strings.size()); i < original_ndim; ++i) {
            final_shape.push_back(original_shape[i]);
            new_strides.push_back(original_strides[i]);
        }
        return std::make_shared<CPUTensor>(memory_block_, new_offset, std::move(final_shape), std::move(new_strides));

    }

    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::transpose(const index_t dim0, const index_t dim1) const {
        const index_t original_ndim = this->get_shape().ndim();
        std::vector<index_t> new_shape = this->get_shape().dims();
        std::vector<index_t> new_strides = this->get_strides();

        // 处理标量（0、1维张量）的情况
        if (original_ndim <= 1) {
            return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
        }

        // 检查维度的有效性
        const index_t adjusted_dim0 = dim0 < 0 ? dim0 + original_ndim : dim0;
        const index_t adjusted_dim1 = dim1 < 0 ? dim1 + original_ndim : dim1;

        if (adjusted_dim0 < 0 || adjusted_dim0 >= original_ndim ||
            adjusted_dim1 < 0 || adjusted_dim1 >= original_ndim) {
            throw std::out_of_range("Invalid dimensions for transpose");
            }

        // 即使维度相同，我们也进行交换操作（虽然实际上不会改变任何东西）
        if (adjusted_dim0 != adjusted_dim1) {
            std::swap(new_shape[adjusted_dim0], new_shape[adjusted_dim1]);
            std::swap(new_strides[adjusted_dim0], new_strides[adjusted_dim1]);
        }

        // 创建新的 CPUTensor，共享原始数据，但其他属性都是新的
        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
    }

    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::permute(const std::vector<index_t>& dims) {
        const index_t original_ndim = this->get_shape().ndim();

        // 检查维度数量是否匹配
        if (static_cast<index_t>(dims.size()) != original_ndim) {
            throw std::invalid_argument(Utils::Format("number of dimensions in the tensor input does not match the length of the desired ordering of "
                                                      "dimensions i.e. input.dim() = %d is not equal to len(dims) = %d", dims.size(), original_ndim));
        }
        //标量和向量保持不变
        if (original_ndim <= 1) {
            return this->shared_from_this();
        }

        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& old_strides = this->get_strides();

        std::vector<index_t> new_shape(original_ndim);
        std::vector<index_t> new_strides(original_ndim);
        std::vector<bool> used(original_ndim, false);

        for (index_t i = 0; i < original_ndim; ++i) {
            index_t dim = dims[i];
            // 处理负数维度
            if (dim < 0) {
                dim += original_ndim;
            }

            // 检查维度的有效性
            if (dim < 0 || dim >= original_ndim) {
                throw std::out_of_range(Utils::Format("Dimension out of range (expected to be in range of [%d, %d], but got %d)", -original_ndim, original_ndim-1, dims[i]));
            }

            // 检查维度是否重复
            if (used[dim]) {
                throw std::invalid_argument("duplicate dims are not allowed.");
            }
            used[dim] = true;

            new_shape[i] = original_shape[dim];
            new_strides[i] = old_strides[dim];
        }

        // 创建新的 CPUTensor，共享原始数据，但其他属性都是新的
        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
    }

    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::flatten() {
        return flatten(0,-1);
    }

    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::flatten(index_t start_dim, index_t end_dim) {
        const index_t original_ndim = this->get_shape().ndim();
        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& origin_strides = get_strides();

        // 处理负数维度
        if (start_dim < 0) start_dim += original_ndim;
        if (end_dim < 0) end_dim += original_ndim;

        if(start_dim == end_dim || original_shape.empty()) {
            return this->shared_from_this();
        }

        // 确保维度在有效范围内
        if (start_dim < 0 || start_dim >= original_ndim || end_dim < 0 || end_dim >= original_ndim || start_dim > end_dim) {
            throw std::out_of_range(Utils::Format("Dimension out of range (expected to be in range of [%d, %d], but got %d)",
                -original_ndim, original_ndim - 1, start_dim == original_ndim ? start_dim : end_dim));
        }

        std::vector<index_t> new_shape;
        std::vector<index_t> new_strides;
        index_t flattened_size = 1;
        const index_t after_flatten_size = original_ndim - end_dim + start_dim;

        // 保留 start_dim 之前的维度
        new_shape.reserve(after_flatten_size);
        new_strides.reserve(after_flatten_size);

        for (index_t i = 0; i < start_dim; ++i) {
            new_shape.push_back(original_shape[i]);
            new_strides.push_back(origin_strides[i]);
        }

        // 计算需要展平的维度的大小
        for (index_t i = start_dim; i <= end_dim; ++i) {
            flattened_size *= original_shape[i];
        }

        new_shape.push_back(flattened_size);
        new_strides.push_back(1);

        // 添加 end_dim 之后的维度
        for (index_t i = end_dim + 1; i < original_ndim; ++i) {
            new_shape.push_back(original_shape[i]);
            new_strides.push_back(origin_strides[i]);
        }

        // 判断需要合并的维度里面有没有非连续的
        if (this->is_contiguous_in_range(start_dim, end_dim)) {
            return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
        }

        // 如果不是连续的，调用 clone()
        auto new_contiguous_tensor = std::dynamic_pointer_cast<CPUTensor>(this->clone());
        return std::make_shared<CPUTensor>(new_contiguous_tensor->memory_block_, new_contiguous_tensor->offset_, std::move(new_shape));
    }


    template <typename ScalarType>
    [[nodiscard]] std::shared_ptr<TensorBase> CPUTensor<ScalarType>::repeat(const std::vector<index_t>& repeats) const {
        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const index_t original_ndim = this->get_shape().ndim();
        const auto repeats_size = static_cast<index_t>(repeats.size());
        const auto& original_strides = this->get_strides();

        std::vector<index_t> new_shape;
        new_shape.reserve(repeats_size);

        if (original_shape.empty())
            throw std::invalid_argument("Cannot repeat a scalar (0-dimensional tensor)");

        if (repeats_size < original_ndim)
            throw std::invalid_argument("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

        const index_t prepend_dims = repeats_size > original_ndim ? repeats_size - original_ndim : 0;

        for (index_t i = 0; i < repeats_size; ++i) {
            if (repeats[i] == 0)
                throw std::invalid_argument(Utils::Format("Trying to create tensor with zero dimension at index %d", i));
            if (repeats[i] < 0)
                throw std::invalid_argument(Utils::Format("Trying to create tensor with negative dimension %d", repeats[i]));

            if (i < prepend_dims) {
                new_shape.push_back(repeats[i]);
            } else {
                new_shape.push_back(repeats[i] * original_shape[i - prepend_dims]);
            }
        }

        auto dst_tensor = std::make_shared<CPUTensor>(new_shape);
        const ScalarType* src_data = this->data();
        ScalarType* dst_data = dst_tensor->mutable_data();

        // 计算外部循环的维度
        index_t outer_dim = 1;
        for (index_t d = 0; d <  repeats_size - original_ndim; ++d) {
            outer_dim *= new_shape[d];
        }
        const index_t inner_dim = outer_dim > 0 ? dst_tensor->size() / outer_dim : 0;

        #pragma omp parallel
        for (index_t i = 0; i < outer_dim; ++i) {
            const index_t dst_offset = i * inner_dim;

            for (index_t j = 0; j < inner_dim; ++j) {
                index_t idx = j;
                index_t src_offset = this->offset_;

                for (index_t k = original_ndim - 1; k >= 0; --k) {
                    index_t src_idx = (idx % new_shape[k + prepend_dims]) % original_shape[k];
                    idx /= new_shape[k + prepend_dims];
                    src_offset += src_idx * original_strides[k];
                }

                dst_data[dst_offset + j] = src_data[src_offset];
            }
        }
        return dst_tensor;
    }

    template <typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::contiguous() {
        // 如果已经是连续的，直接返回 this 的 shared_ptr
        if (this->is_contiguous()) {
            return this->shared_from_this();
        }
        // 如果不是连续的，调用 clone()
        return this->clone();
    }

    template <typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::clone() const {
        const index_t original_ndim = this->shape.ndim();
        const std::vector<index_t> original_shape = this->shape.dims();
        const std::vector<index_t>& original_strides = this->get_strides();
        const ScalarType* src_data = this->data();

        // 处理 0 维度（标量）的情况
        if (original_ndim == 0) {
            auto dst_tensor = std::make_shared<CPUTensor>(Shape{});
            dst_tensor->mutable_data()[0] = src_data[this->offset_];
            return dst_tensor;
        }

        auto dst_tensor = std::make_shared<CPUTensor>(Shape{original_shape});
        ScalarType* dst_data = dst_tensor->mutable_data();
        const auto& dst_strides = dst_tensor->get_strides();

        // 快速路径：如果源张量是连续的，直接进行整体复制
        if (this->is_contiguous()) {
            std::memcpy(dst_data, src_data + this->offset_, this->size() * sizeof(ScalarType));
            return dst_tensor;
        }

        // 计算外部循环的维度
        index_t outer_dim = 1;
        for (index_t d = 0; d < original_ndim - 1; ++d) {
            outer_dim *= original_shape[d];
        }

        const index_t copy_size = original_shape.back();
        const index_t src_inc_step = original_strides[original_ndim-1];

        for (index_t i = 0; i < outer_dim; ++i) {
            index_t src_offset = this->offset_;
            index_t dst_offset = 0;
            index_t idx = i;

            for (index_t j = std::max(original_ndim - 2, static_cast<index_t>(0)); j >= 0; --j) {
                index_t _idx_ = idx % original_shape[j];
                idx /= original_shape[j];
                src_offset += _idx_ * original_strides[j];
                dst_offset += _idx_ * dst_strides[j];
            }

            //blas 负索引 计算方式还是按照 开始位置 和我们通过末尾方式实现不同 需要转换下
            index_t src_blas_offset = src_inc_step < 0 ? src_offset + (copy_size - 1) * src_inc_step : src_offset;
            if constexpr (std::is_same_v<ScalarType, float>) {
                cblas_scopy(copy_size, src_data + src_blas_offset, src_inc_step, dst_data + dst_offset, 1);
            } else if constexpr (std::is_same_v<ScalarType, double>) {
                cblas_dcopy(copy_size, src_data + src_blas_offset, src_inc_step, dst_data + dst_offset, 1);
            } else {
                // 对于其他类型，手动复制以确保正确处理不连续的情况
                for (index_t j = 0; j < copy_size; ++j) {
                    dst_data[dst_offset + j] = src_data[src_offset + j * src_inc_step];
                }
            }
        }

        return dst_tensor;
    }


    template <typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::view(const std::vector<index_t>& new_shape) const {
        // 检查张量是否连续
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot perform view on non-contiguous tensor");
        }
        const std::vector<index_t> original_shape = this->get_shape().dims();
        const index_t total_size = this->get_shape().total_size();
        const auto new_dims = static_cast<index_t>(new_shape.size());

        // 处理自动计算维度
        index_t new_total_size = 1;
        index_t dynamic_dim = -1;
        std::vector<index_t> actual_shape;
        actual_shape.reserve(new_dims);

        for (index_t i = 0; i < new_dims; ++i) {
            if (index_t dim = new_shape[i]; dim == -1) {
                if (dynamic_dim != -1) {
                    throw std::invalid_argument("Only one dimension can be -1 in view");
                }
                dynamic_dim = i;
                actual_shape.push_back(0);  // 占位，稍后填充
            } else if (dim < 0) {
                throw std::invalid_argument("Invalid negative dimension in view");
            } else {
                if (dim == 0) dim = original_shape[i];
                new_total_size *= dim;
                actual_shape.push_back(dim);
            }
        }

        // 计算动态维度
        if (dynamic_dim != -1) {
            if (total_size % new_total_size != 0) {
                throw std::invalid_argument("New shape is not compatible with the number of elements");
            }
            actual_shape[dynamic_dim] = total_size / new_total_size;
            new_total_size *= actual_shape[dynamic_dim];
        }

        if (new_total_size != total_size) {
            throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
        }

        // 返回新的张量视图，保持原始偏移
        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(actual_shape));
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::unsqueeze(const index_t dim) const {
        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& original_strides = this->get_strides();
        const index_t original_ndim = this->get_shape().ndim();

        std::vector<index_t> new_shape;
        std::vector<index_t> new_strides;

        new_shape.reserve(original_ndim + 1);
        new_strides.reserve(original_ndim + 1);

        // 处理负数维度
        const index_t adjusted_dim = dim < 0 ? original_ndim + dim + 1 : dim;
        if (adjusted_dim < 0 || adjusted_dim > original_ndim) {
            throw std::out_of_range(
                Utils::Format("Dimension out of range (expected to be in range of [%d, %d], but got %d)",
                    (original_shape.size() + 1) * -1, original_shape.size() -1, dim));
        }
        //处理下 维度 超过
        for (index_t i = 0; i < original_ndim + 1; ++i) {
            if (i == adjusted_dim) {
                new_shape.push_back(1);
                new_strides.push_back(1);
            }

            if (i < original_ndim) {
                new_shape.push_back(original_shape[i]);
                new_strides.push_back(original_strides[i]);
            }
        }

        new_strides = Utils::compute_strides_with_origin(new_shape, new_strides);

        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::squeeze() const {
        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& original_strides = this->get_strides();
        const index_t original_ndim = this->get_shape().ndim();

        std::vector<index_t> new_shape;
        std::vector<index_t> new_strides;

        new_shape.reserve(original_ndim);
        new_strides.reserve(original_ndim);

        for (index_t i = 0; i < original_ndim; ++i) {
            // 如果维度不是1或者是广播维度，则保留
            if (original_shape[i] != 1 || original_strides[i] == 0) {
                new_shape.push_back(original_shape[i]);
                new_strides.push_back(original_strides[i]);
            }
        }

        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
    }

    template<typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::squeeze(const index_t dim) const {
        const std::vector<index_t>& original_shape = this->get_shape().dims();
        const std::vector<index_t>& original_strides = this->get_strides();
        const index_t original_ndim = this->get_shape().ndim();

        std::vector<index_t> new_shape;
        std::vector<index_t> new_strides;

        new_shape.reserve(original_ndim);
        new_strides.reserve(original_ndim);

        const index_t adjusted_dim = dim < 0 ? original_ndim + dim : dim;
        if (adjusted_dim < 0 || adjusted_dim >= original_ndim) {
            throw std::out_of_range(
                Utils::Format("Dimension out of range (expected to be in range of [%d, %d], but got %d)",
                    new_shape.size() * -1,  new_shape.size() -1, dim));
        }
        for (index_t i = 0; i < original_ndim; ++i) {
            // 如果 不是当前维度 维度不是1 或者广播 维度 都去不掉
            if (i != adjusted_dim || original_shape[i] != 1 || original_strides[i] == 0) {
                new_shape.push_back(original_shape[i]);
                new_strides.push_back(original_strides[i]);
            }
        }

        // 使用新函数重新计算strides
        new_strides = Utils::compute_strides_with_origin(new_shape, new_strides);

        return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(new_shape), std::move(new_strides));
    }

    template <typename ScalarType>
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::expand(const std::vector<index_t>& new_shape) const {
            const std::vector<index_t>& original_shape = this->get_shape().dims();
            const std::vector<index_t>& original_strides = this->get_strides();
            const index_t original_ndim = this->get_shape().ndim();
            const auto expand_size = static_cast<index_t>(new_shape.size());
            // 计算左侧新增的维度数量
            const index_t left_padding = expand_size - original_ndim;

            // 检查新shape的维度数是否小于当前shape
            if (left_padding < 0) {
                throw std::invalid_argument("New shape must have at least as many dimensions as the current shape");
            }

            std::vector<index_t> actual_new_shape(expand_size);
            std::vector<index_t> new_strides(expand_size);

            // 处理左侧新增的维度
            for (index_t i = 0; i < left_padding; ++i) {
                if (new_shape[i] == -1) {
                    actual_new_shape[i] = 1;  // 对于新增维度，-1 被解释为 1
                }else {
                    actual_new_shape[i] = new_shape[i];
                }
                new_strides[i] = 0;  // 新增维度的stride为0
            }

            // 处理原有维度
            for (index_t i = 0; i < original_ndim; ++i) {
                if (const index_t new_index = i + left_padding;
                    new_shape[new_index] == -1 || new_shape[new_index] == original_shape[i]) {
                    actual_new_shape[new_index] = original_shape[i];
                    new_strides[new_index] = original_strides[i];
                } else if (original_shape[i] == 1) {
                    actual_new_shape[new_index] = new_shape[new_index];
                    new_strides[new_index] = 0;
                }else {
                    throw std::invalid_argument("Invalid expansion: new size must be -1, or match current size, or current size must be 1");
                }
            }

            return std::make_shared<CPUTensor>(memory_block_, this->offset_, std::move(actual_new_shape), std::move(new_strides));
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::arange(const ScalarType start, const ScalarType end, const ScalarType step) {
        if (step == 0) {
            throw std::invalid_argument("step must be non-zero");
        }
        if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
            throw std::invalid_argument("invalid range");
        }
        const auto size = static_cast<index_t>(std::ceil((end - start) / step));
        auto tensor = vector(size);
        CPUTensorOps<ScalarType>::getInstance().arange(*tensor, start, step);
        return tensor;
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::scalar(const ScalarType value) {
        std::vector<index_t> shape = {};
        auto tensor = std::make_shared<CPUTensor>(std::move(shape));
        tensor->mutable_data()[0] = value;
        return tensor;
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::vector(const index_t size) {
        std::vector<index_t> shape = {size};
        return std::make_shared<CPUTensor>(std::move(shape));
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::vector(const std::string& tensor_elements)
    {
        std::vector<ScalarType> elements;
        size_t start = 0, end = 0;
        while ((end = tensor_elements.find(',', start)) != std::string::npos) {
            std::string element = tensor_elements.substr(start, end - start);
            try {
                elements.push_back(static_cast<ScalarType>(std::stoi(element)));
            } catch (const std::exception& e) {
                BREEZE_ASSERT(false, "Failed to convert element to integer");
            }
            start = end + 1;
        }
        // 处理最后一个元素
        std::string lastElement = tensor_elements.substr(start);
        try {
            elements.push_back(static_cast<ScalarType>(std::stoi(lastElement)));
        } catch (const std::exception& e) {
            BREEZE_ASSERT(false, "Failed to convert last element to integer");
        }
        BREEZE_ASSERT(!elements.empty(), "Tensor elements are empty");

        auto shape = {static_cast<index_t>(elements.size())};
        return std::make_shared<CPUTensor>(elements, shape);
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::stack(const std::vector<Tensor<ScalarType>*>& tensors, index_t dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("No tensors provided for stacking");
        }

        const index_t original_ndim = tensors[0]->get_shape().ndim();
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (const auto& tensor = tensors[i]; tensor->get_shape() != tensors[0]->get_shape()) {
                throw std::invalid_argument(Utils::Format("stack expects each tensor to be equal size, but got %s at entry %d and %s at entry %d",
                                                          tensors[0]->get_shape().dims_str().c_str(), 0, tensor->get_shape().dims_str().c_str(), i));
            }
        }

        if (dim < 0) {dim += original_ndim + 1;} // +1 because we're adding a new dimension
        if (dim < 0 || dim > original_ndim) {
            throw std::invalid_argument(Utils::Format("Dimension out of range (expected to be in range of [%d, %d],"
                                                      " but got %d)", -original_ndim, original_ndim - 1, dim));
        }

        // 计算新的形状
        std::vector<index_t> new_shape = tensors[0]->get_shape().dims();
        new_shape.insert(new_shape.begin() + dim, tensors.size());
        // 创建结果张量
        auto result = std::make_shared<CPUTensor>(std::move(new_shape));
        // 在视图上执行cat操作
        combine_tensors_out(tensors, dim, result.get());

        return result;
    }

    template <typename ScalarType>
    std::shared_ptr<CPUTensor<ScalarType>> CPUTensor<ScalarType>::cat(const std::vector<Tensor<ScalarType>*>& tensors, index_t dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("No tensors provided for concatenation");
        }

        if (tensors.size() == 1) {
            return std::dynamic_pointer_cast<CPUTensor<ScalarType>>(tensors[0]->contiguous());
        }

        index_t original_ndim = tensors[0]->get_shape().ndim();
        std::vector<index_t> original_shape = tensors[0]->get_shape().dims();

        if (dim < 0) {dim += original_ndim;}
        if (dim < 0 || dim >= original_ndim) {
            throw std::invalid_argument(Utils::Format("Dimension out of range (expected to be in range of [%d, %d], but got %d)", -original_ndim, original_ndim-1, dim));
        }

        index_t concat_dim_size = original_shape[dim];

        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i]->get_shape().ndim() == 0) {
                throw std::invalid_argument(Utils::Format("zero-dimensional tensor (at position %d) cannot be concatenated", i));
            }

            if (tensors[i]->get_shape().ndim() != original_ndim) {
                throw std::invalid_argument(Utils::Format("Tensors must have same number of dimensions: got %d and %d", tensors[i]->get_shape().ndim(), original_ndim));
            }

            concat_dim_size += tensors[i]->get_shape().dims()[dim];
            for (index_t j = 0; j < original_ndim; ++j) {
                if (j != dim && tensors[i]->get_shape().dims()[j] != original_shape[j]) {
                    throw std::invalid_argument(
                        Utils::Format("Sizes of tensors must match except in dimension %d. Expected size %d but got size %d "
                                      "for tensor number %d in the list.", dim, original_shape[j], tensors[i]->get_shape().dims()[j], i));
                }
            }
        }

        original_shape[dim] = concat_dim_size;
        auto result = std::make_shared<CPUTensor<ScalarType>>(std::move(original_shape));
        combine_tensors_out(tensors, dim, result.get());
        return result;
    }

    template <typename ScalarType>
    void CPUTensor<ScalarType>::combine_tensors_out(const std::vector<Tensor<ScalarType>*>& tensors, const index_t dim, CPUTensor<ScalarType>* result) {
        if (tensors.empty()) {
            throw std::invalid_argument("No tensors provided for concatenation");
        }
        const index_t original_ndim = tensors[0]->get_shape().ndim();
        const std::vector<index_t>& new_shape = result->get_shape().dims();
        ScalarType* result_data = result->mutable_data();

        // 计算外层循环的次数和步长
        std::vector<index_t> outer_steps(dim);
        index_t outer_iterations = 1;
        for (index_t i = 0; i < dim; ++i) {
            outer_steps[i] = outer_iterations;
            outer_iterations *= tensors[0]->get_shape().dims()[i];
        }

        index_t block_size = 1;
        for (index_t s = dim + 1; s < original_ndim - 1; ++s) {
            block_size *= tensors[0]->get_shape().dims()[s];
        }

        index_t result_offset = 0;
        for (index_t i = 0; i < outer_iterations; ++i) {
            for (const auto& tensor : tensors) {
                // 最后一维的大小，如果 dim 是最后一个维度也就是按列，返回 1
                index_t last_dim_size = dim >= original_ndim ? 1 : tensor->get_shape().dims()[original_ndim - 1];

                auto _block_size = block_size;
                // 计算每次复制的块大小（不包括最后一维）
                const auto src_tensor = static_cast<CPUTensor*>(tensor);
                index_t src_offset = src_tensor->offset_;
                const ScalarType* src_data = src_tensor->data();
                const std::vector<index_t>& src_strides = src_tensor->get_strides();
                const std::vector<index_t>& src_shape = src_tensor->get_shape().dims();

                if (src_shape.size() >= 2 && dim <= src_tensor->get_shape().ndim() - 2) {
                    _block_size *= src_shape[dim];
                }

                index_t remaining = i;
                for (index_t d = dim - 1; d >= 0 && remaining > 0; --d) {
                    src_offset += (remaining % src_shape[d]) * src_strides[d];
                    remaining /= src_shape[d];
                }

                if (src_tensor->is_contiguous() && result->is_contiguous()) {
                    // 使用 memcpy 进行连续内存的复制
                    const index_t copy_size = _block_size * last_dim_size;
                    std::memcpy(result_data + result_offset, src_data + src_offset, copy_size * sizeof(ScalarType));
                    result_offset += copy_size;
                } else {
                    // 对于非连续存储，我们使用逐元素复制
                    for (index_t j = 0; j < _block_size; ++j) {
                        index_t current_src_offset = src_offset, dst_offset = result_offset;
                        remaining = j;
                        const auto current_dm = new_shape.size() - src_shape.size();

                        for (index_t d = new_shape.size() - 2; d >= dim; --d) {
                            dst_offset += (remaining % new_shape[d]) * result->get_strides()[d];
                            if (d != dim)
                                current_src_offset += (remaining % src_shape[d-current_dm]) * src_strides[d-current_dm];
                            remaining /= new_shape[d];
                        }

                        const index_t src_inc_step = src_strides[original_ndim-1];
                        index_t src_blas_offset = src_inc_step < 0 ? current_src_offset + (last_dim_size - 1) * src_inc_step : current_src_offset;

                        if constexpr (std::is_same_v<ScalarType, float>) {
                            cblas_scopy(last_dim_size, src_data + src_blas_offset, src_inc_step,
                                        result_data + dst_offset, 1);
                        } else if constexpr (std::is_same_v<ScalarType, double>) {
                            cblas_dcopy(last_dim_size, src_data + src_blas_offset, src_inc_step,
                                        result_data + dst_offset, 1);
                        } else {
                            for (index_t k = 0; k < last_dim_size; ++k) {
                                result_data[dst_offset + k] = src_data[current_src_offset + k * src_inc_step];
                            }
                        }
                    }
                    result_offset += _block_size * last_dim_size;
                }
            }
        }
    }
    // Explicit instantiation for common types
    template class CPUTensor<float>;
    template class CPUTensor<double>;

} // namespace Breeze