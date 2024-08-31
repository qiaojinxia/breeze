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

    template<typename T>
    class Tensor;

    template<typename T>
    class CPUTensor;

    template<typename T>
    class TensorIterator {
    public:
        struct OperandInfo {
            T* data;
            std::vector<index_t> strides ={};
            index_t offset = 0;
            bool is_output{false};
            bool is_read_write{false};
        };

        TensorIterator() = default;

        static TensorIterator binary_op(Tensor<T>& out, const Tensor<T>& a, const Tensor<T>& b) {
            TensorIterator iter;
            iter.add_input(a);
            iter.add_input(b);
            iter.add_output(out);
            iter.build();
            return iter;
        }

        void add_output(Tensor<T>& t) {
            out_tensor = &t;
        }

        void add_input(const Tensor<T>& t) {
            operands_.push_back(OperandInfo{const_cast<T*>(t.data()), t.get_strides(), t.get_offset(), false, false});
            tensors_.push_back(const_cast<Tensor<T>*>(&t));
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

            for (auto& op : operands_) {
                op.strides = expand_strides(op.strides, shape_);
            }

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
            }
            // } else if (is_inner_contiguous_) {
            //     inner_contiguous_kernel_vec(scalar_op, vector_op);
            // } else {
            //     strided_kernel_vec(scalar_op, vector_op);
            // }
        }

    private:
        Tensor<T>* out_tensor;
        std::vector<OperandInfo> operands_;
        std::vector<Tensor<T>*> tensors_;
        std::vector<index_t> shape_{};
        bool is_contiguous_{};
        bool is_inner_contiguous_{};
        size_t common_dim_{};

        template<typename Func>
        void contiguous_for_each(Func& func) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
            #pragma omp parallel for schedule(dynamic, 64)
            for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
                const index_t end = std::min(i + GRAIN_SIZE, numel);
                call_function(func, i, end);
            }
        }

        template<typename Func>
        void inner_contiguous_for_each(Func& func) {
            const index_t inner_dim = shape_.back();
            const index_t outer_dim = std::accumulate(shape_.begin(), shape_.end() - 1, 1LL, std::multiplies<>());

            #pragma omp parallel
            {
                std::vector<index_t> counter(shape_.size() - 1, 0);
                std::vector<T*> data_ptrs(operands_.size());

                #pragma omp for schedule(dynamic, 64)
                for (index_t i = 0; i < outer_dim; ++i) {
                    index_to_counter(i, counter);
                    for (size_t op = 0; op < operands_.size(); ++op) {
                        data_ptrs[op] = operands_[op].data + compute_offset(counter, operands_[op].strides);
                    }
                    func(data_ptrs.data(), inner_dim);
                }
            }
        }

        template<typename Func>
        void strided_for_each(Func& func) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());

            #pragma omp parallel
            {
                std::vector<index_t> counter(shape_.size(), 0);
                std::vector<T*> data_ptrs(operands_.size());

                #pragma omp for schedule(dynamic, 64)
                for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
                    const index_t end = std::min(i + GRAIN_SIZE, numel);
                    index_to_counter(i, counter);
                    for (index_t j = i; j < end; ++j) {
                        for (size_t op = 0; op < operands_.size(); ++op) {
                            data_ptrs[op] = operands_[op].data + compute_offset(counter, operands_[op].strides);
                        }
                        func(data_ptrs.data(), 1);
                        increment_counter(counter);
                    }
                }
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void contiguous_kernel_vec(ScalarOp& scalar_op, VectorOp& vector_op) {
            // Calculate the total number of elements based on the shape
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());

            #pragma omp parallel for schedule(dynamic, 64)
            for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
                const index_t end = std::min(i + GRAIN_SIZE, numel);

                // Prepare the data pointers for scalar and vector operations
                std::array<T*, function_traits<ScalarOp>::arity> data_ptrs_scalar;
                std::array<T*, function_traits<VectorOp>::arity> data_ptrs_vector;

                // Initialize pointers for scalar and vector operations
                for (size_t j = 0; j < data_ptrs_scalar.size(); ++j) {
                    data_ptrs_scalar[j] = operands_[j].data + i;
                }

                for (size_t j = 0; j < data_ptrs_vector.size(); ++j) {
                    data_ptrs_vector[j] = operands_[j].data + i;
                }

                // Assume output_ptr is the last operand for both scalar and vector operations
                T* output_ptr = operands_.back().data + i;

                // Call the operation loop with the correct data pointers and output pointer
                op_loop(scalar_op, vector_op, data_ptrs_scalar, data_ptrs_vector, output_ptr, end - i);
            }
        }


        template<typename Op>
        void inner_contiguous_kernel_vec(Op& op) {
            const index_t inner_dim = shape_.back();
            const index_t outer_dim = std::accumulate(shape_.begin(), shape_.end() - 1, 1LL, std::multiplies<>());
            #pragma omp parallel
            {
                std::vector<index_t> counter(shape_.size() - 1, 0);
                #pragma omp for schedule(dynamic, 64)
                for (index_t i = 0; i < outer_dim; ++i) {
                    index_to_counter(i, counter);
                    call_function_vec(op, counter, inner_dim);
                }
            }
        }

        template<typename Op>
        void strided_kernel_vec(Op& op) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());

            #pragma omp parallel
            {
                std::vector<index_t> counter(shape_.size(), 0);

                #pragma omp for schedule(dynamic, 64)
                for (index_t i = 0; i < numel; i += GRAIN_SIZE) {
                    const index_t end = std::min(i + GRAIN_SIZE, numel);
                    index_to_counter(i, counter);
                    call_function_vec(op, counter, end - i);
                }
            }
        }

        template<typename Func>
        void call_function(Func& func, const index_t start, const index_t end) {
            std::vector<T*> data_ptrs(operands_.size());
            for (size_t i = 0; i < operands_.size(); ++i) {
                data_ptrs[i] = operands_[i].data + start * operands_[i].strides.back();
            }
            func(data_ptrs.data(), end - start);
        }

        template<typename Op>
        void call_function_vec(Op& op, const index_t start, const index_t size) {
            constexpr size_t num_operands = function_traits<Op>::arity - 1;
            std::array<T*, num_operands> data_ptrs;
            for (size_t i = 0; i < num_operands; ++i) {
                data_ptrs[i] = operands_[i].data + start * operands_[i].strides.back() + operands_[i].offset;
            }
            // op_loop(op, data_ptrs, size);
        }

        template<typename Op>
        void call_function_vec(Op& op, const std::vector<index_t>& counter, const index_t size) {
            constexpr size_t num_operands = function_traits<Op>::arity - 1;
            std::array<T*, num_operands> data_ptrs;
            for (size_t i = 0; i < num_operands; ++i) {
                data_ptrs[i] = operands_[i].data + compute_offset(counter, operands_[i].strides);
            }
            // op_loop(op, data_ptrs, size);
        }


        template< typename VectorOp, size_t... I>
        void vectorized_loop_impl(VectorOp& op, const std::array<T*, function_traits<VectorOp>::arity>& data_ptrs,
                          T* output_ptr, const index_t size, std::index_sequence<I...>) {
            for (index_t i = 0; i + Vectorized<T>::size() < size; i += Vectorized<T>::size()) {
                // Use SIMD-enabled function if the block size is large enough and type is SIMD-capable
                op(output_ptr + i, Vectorized<T>::loadu(data_ptrs[I] + i)...);
            }
        }

        template<typename ScalarOp, size_t... I>
        void scalar_loop_impl(ScalarOp& op, const std::array<T*, function_traits<ScalarOp>::arity>& data_ptrs,
                      T* output_ptr, const index_t size, std::index_sequence<I...>) {
            auto begin = size - size % Vectorized<T>::size();
            for (index_t i = begin; i < size; i += 1) {
                    // Process each element in the block using scalar operations
                    op(output_ptr + i, data_ptrs[I]+ i...);
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void op_loop(ScalarOp& scalar_op, VectorOp& vector_op,const std::array<T*, function_traits<ScalarOp>::arity>& data_ptrs_scalar,
                const std::array<T*, function_traits<VectorOp>::arity>& data_ptrs_vector, T* output_ptr, const index_t size) {
            constexpr size_t num_operands = function_traits<VectorOp>::arity -1;
            vectorized_loop_impl(vector_op, data_ptrs_vector, output_ptr, size, std::make_index_sequence<num_operands>{});
            // If not vectorized, fall back to the scalar implementation
            scalar_loop_impl(scalar_op, data_ptrs_scalar, output_ptr, size, std::make_index_sequence<num_operands>{});

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