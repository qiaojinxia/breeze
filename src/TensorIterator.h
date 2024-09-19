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
#include "ScalarType.h"
#include "TensorIteratorConfig.h"

namespace Breeze {
    static constexpr index_t GRAIN_SIZE = 131072;
    static constexpr index_t CACHE_LINE_SIZE = 64;  // Typical cache line size

    // 辅助模板来获取类型包中的最后一个类型
    template<typename... ScalarTypes>
    struct last_type;

    template<typename ScalarType>
    struct last_type<ScalarType> {
        using type = ScalarType;
    };

    template<typename ScalarType, typename... ScalarTypes>
    struct last_type<ScalarType, ScalarTypes...> : last_type<ScalarTypes...> {};

    template<typename... ScalarTypes>
    using LastScalarType = typename last_type<ScalarTypes...>::type;

    template<typename ScalarType>
    class Tensor;

    template<typename... ScalarTypes>
    class TensorIterator {
    public:
        using OutputScalarType = LastScalarType<ScalarTypes...>;
        using ResultScalarType = typename last_type<ScalarTypes...>::type;

        //输出类型 的大小
        static constexpr size_t ResultTypeSize = sizeof(OutputScalarType);
        static constexpr size_t ScalarTypesSize = sizeof ...(ScalarTypes);
        static constexpr size_t OptPutIndex  = ScalarTypesSize - 1;

        struct OperandInfo {
            char* data{};
            std::vector<index_t> strides ={};
            std::vector<index_t> strides_bytes ={};
            index_t begin_offset = 0;
            bool is_output{false};
            bool is_read_write{false};
            ScalarType ScalarType{};

            OperandInfo()= default;

            template <typename T>
            OperandInfo(T* data, const std::vector<index_t>& strides, const index_t offset,
                const bool is_output, const bool is_read_write)
            : data(reinterpret_cast<char*>(data)),
            strides(strides),
            begin_offset(offset),
            is_output(is_output),
            is_read_write(is_read_write),
            ScalarType(TypeToScalarType<T>::value) {}
        };

        TensorIterator() = default;

        explicit TensorIterator(const TensorIteratorConfig& config): operands_()  {
            // 使用 config 初始化 TensorIterator
            config_ = config;
        }

        template<typename ScalarT1, typename ScalarT2>
        static TensorIterator<ScalarT1, ScalarT2, typename BinaryOpResultType<ScalarT1, ScalarT2>::type>
        binary_op(Tensor<typename BinaryOpResultType<ScalarT1, ScalarT2>::type>& out,
          const Tensor<ScalarT1>& a, const Tensor<ScalarT2>& b,
          const TensorIteratorConfig& config = TensorIteratorConfig::default_config()) {
            using OutputType = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
            auto iter = config.build<ScalarT1, ScalarT2, typename BinaryOpResultType<ScalarT1, ScalarT2>::type>();
            iter.template add_input<0, ScalarT1>(a);
            iter.template add_input<1, ScalarT2>(b);
            iter.template add_output<OutputType>(out);
            iter.build(std::make_index_sequence<3>{});
            return iter;
        }

        template<typename InputScalarType, typename OutputScalarType = InputScalarType>
            static TensorIterator<InputScalarType, OutputScalarType>
            unary_op(Tensor<OutputScalarType>& out, const Tensor<InputScalarType>& input,
         const TensorIteratorConfig& config = TensorIteratorConfig::default_config()) {
            auto iter = config.build<InputScalarType, OutputScalarType>();
            iter.template add_input<0, InputScalarType>(input);
            iter.template add_output<OutputScalarType>(out);
            iter.build(std::make_index_sequence<2>{});
            return iter;
        }

        template<typename ScalarType>
        static TensorIterator<ScalarType> nullary_op(Tensor<ScalarType>& out,
           const TensorIteratorConfig& config = TensorIteratorConfig::default_config()) {
            auto iter = config.build<ScalarType>();
            iter.template add_output<ScalarType>(out);
            iter.build(std::make_index_sequence<1>{});
            return iter;
        }

        template<std::size_t I = 0>
        [[nodiscard]] std::vector<index_t> get_shape_from_tuple() const {
            if constexpr (I == sizeof...(ScalarTypes)) {
                return {};
            } else {
                return std::get<I>(tensors_)->get_shape().dims();
            }
        }

        // 修改 add_output 方法，确保输出在最后一个模板参数位置
        template<typename ScalarType>
        void add_output(Tensor<ScalarType>& t) {
            operands_[OptPutIndex] = OperandInfo(const_cast<ScalarType*>(t.data()), t.get_strides(), t.get_offset(), true, true);
            std::get<OptPutIndex>(tensors_) = static_cast<Tensor<ScalarType>*>(&t);
        }

        // 修改 add_input 方法，确保输入在前面的模板参数位置
        template<size_t I, typename ScalarType>
        void add_input(const Tensor<ScalarType>& t) {
            operands_[I] = OperandInfo(const_cast<ScalarType*>(t.data()), t.get_strides(), t.get_offset(), false, false);
            std::get<I>(tensors_) = const_cast<Tensor<ScalarType>*>(&t);
        }

        template <std::size_t I>
        void expand_strides_for_operand() {
            operands_[I].strides = expand_strides(get_shape_from_tuple<I>(), operands_[I].strides, shape_);
        }

        template <std::size_t I>
        void calc_strides_for_operand() {
            operands_[I].strides_bytes = calc_strides_bytes(operands_[I].ScalarType, operands_[I].strides);
        }

        template <std::size_t... Is>
        void build(std::index_sequence<Is...>) {

            if (config_.check_all_same_dtype_) {
                check_all_same_dtype();
            }

            if (config_.check_all_same_shape_) {
                (check_all_sanme_shape<Is>(), ...);
            }

            if (ScalarTypesSize > 1) shape_ = compute_common_shape(std::make_index_sequence<2>{});

            if (config_.resize_outputs_) {
                if (!shape_.empty() && get_tensor<OptPutIndex>().get_shape().dims() != shape_) {
                    auto new_shape = Shape(std::vector(shape_.begin(), shape_.end()));
                    get_tensor<OptPutIndex>().set_initial_shape(new_shape);
                    operands_[OptPutIndex] = OperandInfo(const_cast<OutputScalarType*>(get_tensor<OptPutIndex>().mutable_data()),
                        get_tensor<OptPutIndex>().get_strides(),
                        get_tensor<OptPutIndex>().get_offset(), true, true);
                }
            }

            (expand_strides_for_operand<Is>(), ...);
            (calc_strides_for_operand<Is>(), ...);

            if (config_.enforce_safe_casting_to_output_) {
                check_safe_to_output();
            }

            if (shape_.empty()) {
                shape_ = std::vector<index_t>(get_tensor<OptPutIndex>().get_shape().dims().begin(),
                get_tensor<OptPutIndex>().get_shape().dims().end());
            }

            is_contiguous_ = true;
            common_dim_ = 0;
            for (const auto& op : operands_) {
                is_contiguous_ &= is_contiguous(op.strides, shape_);
                common_dim_ = std::max(common_dim_, shape_.size() - op.strides.size());
            }

            is_inner_contiguous_ = true;
            for (const auto& op : operands_) {
                if (!op.strides.empty() && op.strides.back() != 1) {
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
                strided_kernel_vec(scalar_op);
            }
        }

    private:
        std::array<OperandInfo, sizeof ...(ScalarTypes)> operands_ ;
        std::tuple<Tensor<ScalarTypes>*...> tensors_;
        std::vector<index_t> shape_{};
        bool is_contiguous_{};
        bool is_inner_contiguous_{};
        size_t common_dim_{};
        TensorIteratorConfig config_{};

        template<size_t I>
        auto& get_tensor() {
            return *std::get<I>(tensors_);
        }

        template<size_t I>
        [[nodiscard]] index_t calculate_data_size() const {
            index_t size = 1;
            const auto& op = get_operand<I>();
            for (size_t i = 0; i < shape_.size(); ++i) {
                size += (shape_[i] - 1) * op.strides[i];
            }
            return size;
        }

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
            std::array<char*, ScalarTypesSize> data_ptrs;
            std::array<index_t, ScalarTypesSize> data_strides;
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
                    for (size_t op = 0; op < ScalarTypesSize; ++op) {
                        data_ptrs[op] = operands_[op].data + ResultTypeSize * operands_[op].begin_offset +
                            compute_offset(counter, operands_[op].strides_bytes);
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
            std::array<char*, ScalarTypesSize> data_ptrs;
            std::array<index_t, ScalarTypesSize> data_strides;
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
                        for (size_t op = 0; op < ScalarTypesSize; ++op) {
                            data_ptrs[op] =  operands_[op].data + operands_[op].begin_offset * ResultTypeSize +
                                compute_offset(counter, operands_[op].strides_bytes);
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
            std::array<char*, sizeof...(ScalarTypes)> data_ptrs;
            for (size_t j = 0; j < ScalarTypesSize; ++j) {
                data_ptrs[j] = operands_[j].data + operands_[j].begin_offset * ResultTypeSize;
            }
            constexpr index_t grain_size = GRAIN_SIZE;
            #pragma omp parallel for default(none) \
            shared(numel, scalar_op, vector_op, operands_, data_ptrs, grain_size) \
            schedule(dynamic, 64)
            for (index_t i = 0; i < numel; i += grain_size) {
                const index_t end = std::min(i + GRAIN_SIZE, numel);
                std::array<char*, ScalarTypesSize> local_ptrs = data_ptrs;
                for (size_t j = 0; j < ScalarTypesSize; ++j) {
                    local_ptrs[j] += i * ResultTypeSize;
                }
                char* output_ptr = local_ptrs[OptPutIndex];
                op_loop(scalar_op, vector_op, local_ptrs, output_ptr, end - i);
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void inner_contiguous_kernel_vec(ScalarOp& scalar_op, VectorOp& vector_op) {
            const index_t inner_dim = shape_.back();
            const index_t outer_dim = std::accumulate(shape_.begin(), shape_.end() - 1, 1LL, std::multiplies<>());
            std::array<char*, ScalarTypesSize> data_ptrs;
            for (size_t j = 0; j < ScalarTypesSize; ++j) {
                data_ptrs[j] = operands_[j].data + operands_[j].begin_offset * ResultTypeSize;
            }

            #pragma omp parallel for default(none) \
            shared(outer_dim, scalar_op, vector_op, data_ptrs, shape_, inner_dim, operands_) \
            schedule(dynamic, 64)
            for (index_t i = 0; i < outer_dim; ++i) {
                std::array<char*, ScalarTypesSize> local_ptrs = data_ptrs;
                std::vector<index_t> counter(shape_.size() - 1, 0);
                index_to_counter(i, counter);

                for (size_t j = 0; j < ScalarTypesSize; ++j) {
                    local_ptrs[j] += compute_offset(counter, operands_[j].strides_bytes);
                }

                char* output_ptr = local_ptrs[OptPutIndex];
                op_loop(scalar_op, vector_op, local_ptrs, output_ptr, inner_dim);
            }
        }

        template<typename ScalarOp>
        void strided_kernel_vec(ScalarOp& scalar_op) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<>());
            constexpr index_t grain_size = GRAIN_SIZE;
            std::array<char*, ScalarTypesSize> data_ptrs;

            #pragma omp parallel for default(none) \
            shared(numel, scalar_op, operands_, data_ptrs, grain_size) \
            schedule(dynamic, 64)
            for (index_t i = 0; i < numel; i += grain_size) {
                std::vector<index_t> counter(shape_.size(), 0);
                const index_t end = std::min(i + grain_size, numel);
                index_to_counter(i, counter);

                for (index_t j = i; j < end; ++j) {
                    for (size_t k = 0; k < data_ptrs.size(); ++k) {
                        data_ptrs[k] = operands_[k].data + operands_[k].begin_offset * ResultTypeSize +
                            compute_offset(counter, operands_[k].strides_bytes);
                    }

                    char* output_ptr = data_ptrs[OptPutIndex];
                    scalar_loop_impl(scalar_op, data_ptrs, output_ptr, 1, 1, std::make_index_sequence<ScalarTypesSize - 1>{});
                    increment_counter(counter);
                }
            }
        }

        template<typename Func>
        void call_function(Func& func, const index_t start, const index_t end) {
            std::array<char*, ScalarTypesSize> data_ptrs;
            std::array<index_t, ScalarTypesSize> data_strides;
            for (size_t i = 0; i < ScalarTypesSize; ++i) {
                data_ptrs[i] = operands_[i].data +  start * operands_[i].strides_bytes.back() + operands_[i].begin_offset;
                data_strides[i] = 1;
            }
            func(data_ptrs.data(), data_strides, end - start);
        }

        template<typename VectorOp, size_t... I>
        static void vectorized_loop_impl(VectorOp& op, const std::array<char*, ScalarTypesSize>& data_ptrs,
                              char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<I...>) {
            for (index_t i = 0; i <= size - simd_vector_size; i += simd_vector_size) {
                op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                     Vectorized<ResultScalarType>::loadu(data_ptrs[I] + i * sizeof (ResultScalarType))...);
            }
        }

        template<typename ScalarOp, size_t... I>
        static void scalar_loop_impl(ScalarOp& op, const std::array<char*, ScalarTypesSize> &data_ptrs,
            char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<I...>) {
            const auto begin = size - size % simd_vector_size;
            for (index_t i = begin; i < size; i += 1) {
                //这里 不同类型转换使用默认编译器规则
                op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                   *(reinterpret_cast<ResultScalarType*>(data_ptrs[I]) + i)...);
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void op_loop(ScalarOp& scalar_op, VectorOp& vector_op,
            const std::array<char*, ScalarTypesSize> &data_ptrs, char* output_ptr, const index_t size) {
            constexpr size_t simd_vector_size = Vectorized<OutputScalarType>::size();
            vectorized_loop_impl(vector_op, data_ptrs, output_ptr, size, simd_vector_size, std::make_index_sequence<ScalarTypesSize - 1>{});
            scalar_loop_impl(scalar_op, data_ptrs, output_ptr, size, simd_vector_size, std::make_index_sequence<ScalarTypesSize - 1>{});
        }

        void check_all_same_dtype() const{
            ScalarType first_type = operands_[0].ScalarType;
            for (const auto& op : operands_) {
                if (op.ScalarType != first_type) {
                    throw std::runtime_error("All tensors must have the same dtype.");
                }
            }
        }

        void check_safe_to_output() const {
            check_safe_to_output(std::make_index_sequence<sizeof...(ScalarTypes)>{});
        }

        template<size_t... Indices>
        void check_safe_to_output(std::index_sequence<Indices...>) const {
            constexpr size_t num_operands = sizeof...(ScalarTypes);
            std::array<index_t, num_operands> data_size = {calculate_data_size<Indices>()...};
            // 获取输出张量（最后一个操作数）的数据大小
            const index_t output_size = data_size[num_operands - 1];
            // 使用折叠表达式检查输出张量是否有足够的空间
            if (const bool is_safe = ((Indices == num_operands - 1 || output_size >= data_size[Indices]) && ...);!is_safe) {
                throw std::runtime_error("Output tensor does not have enough memory to store the result.");
            }
        }

        template<std::size_t I>
        void check_all_sanme_shape() const{
            if (I==0) return;
            const auto first_shape = get_shape_from_tuple<0>();
            if (get_shape_from_tuple<I>() != first_shape) {
                    throw std::runtime_error("All tensors must have the same shape when no_broadcast is set.");
            }
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

        static std::vector<index_t> expand_strides(const std::vector<index_t>& input_shape,
                                           const std::vector<index_t>& strides,
                                           const std::vector<index_t>& output_shape) {
            if (output_shape.empty()) {
                return strides;
            }
            std::vector<index_t> result(output_shape.size(), 0);
            const size_t offset = output_shape.size() - input_shape.size();
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (i < strides.size()) {
                    if (input_shape[i] == output_shape[i + offset]) {
                        result[i + offset] = strides[i];
                    } else if (input_shape[i] == 1) {
                        // 处理广播情况
                        result[i + offset] = 0;
                    }
                } else {
                    // 如果 strides 比 input_shape 短，用默认值填充
                    result[i + offset] = (input_shape[i] == output_shape[i + offset]) ? 1 : 0;
                }
            }
            // 处理前面的维度（可能是因为广播而新增的维度）
            for (size_t i = 0; i < offset; ++i) {
                result[i] = 0;  // 新增的维度的stride为0
            }

            // 确保至少有一个非零stride
            if (std::all_of(result.begin(), result.end(), [](const index_t x) { return x == 0; })) {
                result.back() = 1;
            }

            return result;
        }

        // 辅助函数：获取特定索引的操作数
        template<size_t I>
        const auto& get_operand() const {
            return std::get<I>(operands_);
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

        template<std::size_t... Is>
        std::vector<index_t> compute_common_shape(std::index_sequence<Is...>) {
            if constexpr (sizeof...(Is) == 0) {
                return {};
            } else if constexpr (sizeof...(Is) == 1) {
                return get_shape_from_tuple<0>();
            } else {
                return broadcast_shapes(get_shape_from_tuple<Is>()...);
            }
        }

        [[nodiscard]] static index_t compute_offset(const std::vector<index_t>& counter, const std::vector<index_t>& strides_bytes) {
            index_t offset = 0;
            for (size_t i = 0; i < counter.size(); ++i) {
                offset += counter[i] * strides_bytes[i];
            }
            return offset;
        }
    };
}

#endif //TENSORITERATOR_H