#ifndef TENSORITERATOR_H
#define TENSORITERATOR_H

#include "Tensor.h"
#include <vector>
#include <numeric>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include "./platform/VectorizedAvx2.h"

namespace Breeze {

// Helper struct to deduce function traits
template<typename F>
struct function_traits : public function_traits<decltype(&F::operator())> {};

template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    static constexpr size_t arity = sizeof...(Args);
};

// Specialization for function pointers
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    static constexpr size_t arity = sizeof...(Args);
};

// Specialization for std::function
template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
    static constexpr size_t arity = sizeof...(Args);
};

static constexpr index_t GRAIN_SIZE = 131072;
static constexpr index_t CACHE_LINE_SIZE = 64;  // Typical cache line size

template<typename Dtype>
class Tensor;

template<typename Dtype>
class CPUTensor;

template<typename Dtype>
class TensorIterator {
public:
    struct OperandInfo {
        Dtype* data;
        std::vector<index_t> strides ={};
        index_t offset = 0;
        bool is_output{false};
        bool is_read_write{false};
    };

    TensorIterator() = default;

    static TensorIterator binary_op(Tensor<Dtype>& out, const Tensor<Dtype>& a, const Tensor<Dtype>& b) {
        TensorIterator iter;
        iter.add_input(a);
        iter.add_input(b);
        iter.add_output(out);
        iter.build();
        return iter;
    }

    void add_output(Tensor<Dtype>& t) {
        out_tensor = &t;
    }

    void add_input(const Tensor<Dtype>& t) {
        operands_.push_back(OperandInfo{const_cast<Dtype*>(t.data()), t.get_strides(), t.get_offset(), false, false});
        tensors_.push_back(const_cast<Tensor<Dtype>*>(&t));
    }

    void build() {
        shape_ = compute_common_shape();
        if (!shape_.empty() && out_tensor->get_shape().dims() != shape_) {
            auto new_shape = Shape(std::vector<index_t>(shape_.begin(), shape_.end()));
            out_tensor->set_initial_shape(new_shape);
        }

        operands_.push_back(OperandInfo{out_tensor->mutable_data(), out_tensor->get_strides(),
        out_tensor->get_offset(), true, true});
        tensors_.push_back(out_tensor);

        is_contiguous_ = true;
        common_dim_ = 0;
        for (const auto& op : operands_) {
            is_contiguous_ &= is_contiguous(op.strides, shape_);
            common_dim_ = std::max(common_dim_, shape_.size() - op.strides.size());
        }

        // for (auto& op : operands_) {
        //     op.strides = expand_strides(op.strides, shape_);
        // }

        is_inner_contiguous_ = true;
        for (const auto& op : operands_) {
            if (op.strides.back() != 1) {
                is_inner_contiguous_ = false;
                break;
            }
        }
    }

    template<typename Func>
    void for_each(Func func) {
        if (is_contiguous_) {
            contiguous_for_each(func);
        } else if (is_inner_contiguous_) {
            inner_contiguous_for_each(func);
        } else {
            strided_for_each(func);
        }
    }

    template<typename ScalarOp, typename VectorOp>
    void cpu_kernel_vec(ScalarOp scalar_op, VectorOp vector_op) {
        if (is_contiguous_) {
            contiguous_kernel_vec(scalar_op, vector_op);
        } else if (is_inner_contiguous_) {
            inner_contiguous_kernel_vec(scalar_op, vector_op);
        } else {
            strided_kernel_vec(scalar_op, vector_op);
        }
    }

private:
    Tensor<Dtype>* out_tensor;
    std::vector<OperandInfo> operands_;
    std::vector<Tensor<Dtype>*> tensors_;
    std::vector<index_t> shape_{};
    bool is_contiguous_{};
    bool is_inner_contiguous_{};
    size_t common_dim_{};


    template<typename Func>
    void contiguous_for_each(Func& func) {
        const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
        #pragma omp parallel for default(none) \
        shared(numel, func) \
        firstprivate(GRAIN_SIZE) \
        schedule(dynamic, 64)
        for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
            const index_t end = std::min(i + GRAIN_SIZE, numel);
            call_function(func, i, end);
        }
    }

    template<typename Func>
    void inner_contiguous_for_each(Func& func) {
        std::vector<index_t> counter(shape_.size() - 1, 0);
        std::vector<Dtype*> data_ptrs(operands_.size());
        std::vector<index_t> data_strides(operands_.size());
        const index_t inner_dim = shape_.back();
        const index_t outer_dim = std::accumulate(shape_.begin(), shape_.end() - 1, 1LL, std::multiplies<>());

        #pragma omp parallel for default(none) \
        shared(outer_dim, inner_dim, func, operands_, shape_) \
        private(counter, data_ptrs, data_strides) \
        firstprivate(GRAIN_SIZE) \
        schedule(dynamic, 64)
        for (index_t i = 0; i < outer_dim; i += GRAIN_SIZE) {
            const index_t end = std::min(i + GRAIN_SIZE, outer_dim);
            index_to_counter(i, counter);
            for (index_t j = i; j < end; ++j) {
                for (size_t op = 0; op < operands_.size(); ++op) {
                    data_ptrs[op] = operands_[op].data + compute_offset(counter, operands_[op].strides) + operands_[op].offset;
                    data_strides[op] = 1;
                }
                func(data_ptrs.data(), data_strides, inner_dim);
                increment_counter(counter);
            }
        }
    }

    template<typename Func>
    void strided_for_each(Func& func) {
        std::vector<index_t> counter(shape_.size(), 0);
        std::vector<Dtype*> data_ptrs(operands_.size());
        std::vector<index_t> data_strides(operands_.size());
        const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
        #pragma omp parallel for default(none) \
            shared(numel, func, operands_, shape_) \
            private(counter, data_ptrs, data_strides) \
            firstprivate(GRAIN_SIZE)   \
            schedule(dynamic, 64)
        {
            for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
                const index_t end = std::min(i + GRAIN_SIZE, numel);
                index_to_counter(i, counter);
                for (index_t j = i; j < end; ++j) {
                    for (size_t op = 0; op < operands_.size(); ++op) {
                        data_ptrs[op] = operands_[op].data + compute_offset(counter, operands_[op].strides) + operands_[op].offset;
                        data_strides[op] = operands_[op].strides.back();
                    }
                    func(data_ptrs.data(), data_strides, 1);
                    increment_counter(counter);
                }
            }
        }
    }

    template<typename ScalarOp, typename VectorOp>
    void contiguous_kernel_vec(ScalarOp& scalar_op, VectorOp& vector_op) {
        const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
        std::array<Dtype*, std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity)> data_ptrs;

        #pragma omp parallel for default(none) \
        shared(numel, scalar_op, vector_op, operands_, shape_) \
        private(data_ptrs) \
        firstprivate(GRAIN_SIZE) \
        schedule(dynamic, 64)
        for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
            const index_t end = std::min(i + GRAIN_SIZE, numel);
            for (size_t j = 0; j < data_ptrs.size(); ++j) {
                data_ptrs[j] = operands_[j].data + i +  operands_[j].offset;
            }
            Dtype* output_ptr = operands_.back().data + i +  operands_.back().offset;
            op_loop(scalar_op, vector_op, data_ptrs, output_ptr, end - i);
        }
    }

    template<typename ScalarOp, typename VectorOp>
            void inner_contiguous_kernel_vec(ScalarOp& scalar_op, VectorOp& vector_op) {
        const index_t inner_dim = shape_.back();
        const index_t outer_dim = std::accumulate(shape_.begin(), shape_.end() - 1, 1LL, std::multiplies<>());
        std::array<Dtype*, std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity)> data_ptrs;

        #pragma omp parallel for default(none) \
        shared(outer_dim, scalar_op, vector_op, shape_, inner_dim, operands_) \
        private(data_ptrs) \
        schedule(dynamic, 64)
        for (index_t i = 0; i < outer_dim; ++i) {
            std::vector<index_t> counter(shape_.size() - 1, 0);
            index_to_counter(i, counter);

            for (size_t j = 0; j < data_ptrs.size(); ++j) {
                data_ptrs[j] = operands_[j].data + compute_offset(counter, operands_[j].strides) + operands_[j].offset;
            }

            Dtype* output_ptr = operands_.back().data + compute_offset(counter, operands_.back().strides);
            op_loop(scalar_op, vector_op, data_ptrs, output_ptr, inner_dim);
        }
    }

    template<typename ScalarOp, typename VectorOp>
    void strided_kernel_vec(ScalarOp& scalar_op, VectorOp& vector_op) {
        const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
        constexpr index_t grain_size = GRAIN_SIZE;
        std::array<Dtype*, std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity)> data_ptrs;

        for (index_t i = 0; i < numel; i += grain_size) {
            std::vector<index_t> counter(shape_.size(), 0);
            const index_t end = std::min(i + grain_size, numel);
            index_to_counter(i, counter);

            for (index_t j = i; j < end; ++j) {
                for (size_t k = 0; k < data_ptrs.size(); ++k) {
                    data_ptrs[k] = operands_[k].data + compute_offset(counter, operands_[k].strides) + operands_[k].offset;
                }

                Dtype* output_ptr = operands_.back().data + compute_offset(counter, operands_.back().strides) + operands_.back().offset;
                constexpr size_t num_operands = std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity) - 1;
                scalar_loop_impl(scalar_op, data_ptrs, output_ptr, 1, std::make_index_sequence<num_operands>{});
                increment_counter(counter);
            }
        }
    }

    template<typename Func>
    void call_function(Func& func, const index_t start, const index_t end) {
        std::vector<Dtype*> data_ptrs(operands_.size());
        std::vector<index_t> data_strides(operands_.size());
        for (size_t i = 0; i < operands_.size(); ++i) {
            data_ptrs[i] = operands_[i].data + start * operands_[i].strides.back() + operands_[i].offset;
            data_strides[i] = 1;
        }
        func(data_ptrs.data(), data_strides, end - start);
    }

    template< typename VectorOp, size_t... I>
    void vectorized_loop_impl(VectorOp& op, const std::array<Dtype*, function_traits<VectorOp>::arity>& data_ptrs,
                      Dtype* output_ptr, const index_t size, std::index_sequence<I...>) {
        for (index_t i = 0; i + Vectorized<Dtype>::size() <= size; i += Vectorized<Dtype>::size()) {
            // Use SIMD-enabled function if the block size is large enough and type is SIMD-capable
            op(output_ptr + i, Vectorized<Dtype>::loadu(data_ptrs[I] + i)...);
        }
    }

    template<typename ScalarOp, size_t... I>
    void scalar_loop_impl(ScalarOp& op, const std::array<Dtype*, function_traits<ScalarOp>::arity>& data_ptrs,
                  Dtype* output_ptr, const index_t size, std::index_sequence<I...>) {
        auto begin = size - size % Vectorized<Dtype>::size();
        for (index_t i = begin; i < size; i += 1) {
                // Process each element in the block using scalar operations
                op(output_ptr + i, data_ptrs[I]+ i...);
        }
    }

    template<typename ScalarOp, typename VectorOp>
    void op_loop(ScalarOp& scalar_op, VectorOp& vector_op,
                 const std::array<Dtype*, std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity)>& data_ptrs,
                 Dtype* output_ptr, const index_t size) {
        constexpr size_t num_operands = std::max(function_traits<ScalarOp>::arity, function_traits<VectorOp>::arity) - 1;
        vectorized_loop_impl(vector_op, data_ptrs, output_ptr, size, std::make_index_sequence<num_operands>{});
        scalar_loop_impl(scalar_op, data_ptrs, output_ptr, size, std::make_index_sequence<num_operands>{});
    }

    void increment_counter(std::vector<index_t>& counter) const {
        for (index_t i = static_cast<index_t>(counter.size()) - 1; i >= 0; --i) {
            if (++counter[i] == shape_[i]) {
                counter[i] = 0;
            } else {
                break;
            }
        }
    }

    void index_to_counter(index_t index, std::vector<index_t>& counter) const {
        for (index_t i = static_cast<index_t>(counter.size()) - 1; i >= 0; --i) {
            counter[i] = index % shape_[i];
            index /= shape_[i];
        }
    }

    static bool is_contiguous(const std::vector<index_t>& strides, const std::vector<index_t>& shape) {
        index_t expected_stride = 1;
        for (index_t i = static_cast<index_t>(shape.size()) - 1; i >= 0; --i) {
            if (shape[i] == 1) continue;
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }

    static std::vector<index_t> broadcast_shapes(const std::vector<index_t>& a, const std::vector<index_t>& b) {
        std::vector<index_t> result(std::max(a.size(), b.size()));
        auto it_a = a.rbegin();
        auto it_b = b.rbegin();
        auto it_result = result.rbegin();
        while (it_a != a.rend() || it_b != b.rend()) {
            index_t dim_a = (it_a != a.rend()) ? *it_a : 1;
            index_t dim_b = (it_b != b.rend()) ? *it_b : 1;
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::runtime_error("Incompatible shapes for broadcasting");
            }
            *it_result = std::max(dim_a, dim_b);
            if (it_a != a.rend()) ++it_a;
            if (it_b != b.rend()) ++it_b;
            ++it_result;
        }
        return result;
    }

    static std::vector<index_t> expand_strides(const std::vector<index_t>& strides, const std::vector<index_t>& shape) {
        if (shape.empty())
            return strides;
        std::vector<index_t> result(shape.size(), 0);
        const index_t offset = static_cast<index_t>(shape.size()) - 1 - static_cast<index_t>(strides.size());
        for (size_t i = 0; i < strides.size(); ++i) {
            result[i + offset] = (shape[i + offset] == 1) ? 0 : strides[i];
        }
        return result;
    }

    std::vector<index_t> compute_common_shape() {
        if (tensors_.empty()) {
            return out_tensor->get_shape().dims();
        }
        std::vector<index_t> common_shape = tensors_[0]->get_shape().dims();
        for (size_t i = 1; i < tensors_.size(); ++i) {
            common_shape = broadcast_shapes(common_shape, tensors_[i]->get_shape().dims());
        }
        return common_shape;
    }

    [[nodiscard]] static index_t compute_offset(const std::vector<index_t>& counter, const std::vector<index_t>& strides) {
        index_t offset = 0;
        for (size_t i = 0; i < counter.size(); ++i) {
            offset += counter[i] * strides[i];
        }
        return offset;
    }
};
}

#endif //TENSORITERATOR_H