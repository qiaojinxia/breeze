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
    constexpr static index_t  TILE_SIZE = CACHE_LINE_SIZE * 64;

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
        using ResultScalarType = LastScalarType<ScalarTypes...>;
        static constexpr index_t SimdVrctorSize = Vectorized<ResultScalarType>::size();
        //输出类型 的大小
        static constexpr size_t ResultTypeSize = sizeof(ResultScalarType);
        static constexpr size_t ScalarTypesSize = sizeof ...(ScalarTypes);
        static constexpr size_t OptPutIndex  = ScalarTypesSize - 1;
        static constexpr index_t PrefetchElements = CACHE_LINE_SIZE / sizeof(ResultScalarType) / SimdVrctorSize; // 以元素数量计的预读取数量

        struct OperandInfo {
            char* data{};
            std::vector<index_t> strides ={};
            std::vector<index_t> strides_bytes ={};
            bool is_output{false};
            bool is_read_write{false};
            ScalarType ScalarType{};

            OperandInfo()= default;
            template <typename T>
            OperandInfo(T* data, const std::vector<index_t>& strides,
                const bool is_output, const bool is_read_write)
            : data(reinterpret_cast<char*>(data)),
            strides(strides),
            is_output(is_output),
            is_read_write(is_read_write),
            ScalarType(TypeToScalarType<T>::value) {}
        };

        TensorIterator() = default;

        explicit TensorIterator(const TensorIteratorConfig& config): operands_()  {
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


        template<typename InputScalarType, typename OutputScalarType = InputScalarType>
        static TensorIterator<InputScalarType, OutputScalarType>
        reduce_op(Tensor<OutputScalarType>& out, const Tensor<InputScalarType>& input,
           const TensorIteratorConfig& config = TensorIteratorConfig::default_config()) {
            auto iter = config.build<InputScalarType, OutputScalarType>();
            iter.template add_input<0, InputScalarType>(input);
            iter.template add_output<OutputScalarType>(out);
            iter.build(std::make_index_sequence<2>{});
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
            operands_[OptPutIndex] = OperandInfo(const_cast<ScalarType*>(t.data() + t.get_offset()), t.get_strides(), true, true);
            std::get<OptPutIndex>(tensors_) = static_cast<Tensor<ScalarType>*>(&t);
        }

        // 修改 add_input 方法，确保输入在前面的模板参数位置
        template<size_t I, typename ScalarType>
        void add_input(const Tensor<ScalarType>& t) {
            operands_[I] = OperandInfo(const_cast<ScalarType*>(t.data() + t.get_offset()), t.get_strides(), false, false);
            std::get<I>(tensors_) = const_cast<Tensor<ScalarType>*>(&t);
        }

        template <std::size_t I>
        void expand_strides_for_operand() {
            operands_[I].strides = Utils::expand_strides(get_shape_from_tuple<I>(), shape_, operands_[I].strides);
            // 针对 reduce处理 一般如果经过广播后维度变化了 1变成n  1 !=n 就把 步长设置为0
            // 同时我们reduce 也是相当于把输出维度 n 变成 1 这样也会调整成0 所以resize的时候我们手动设置成了1 缩减维度
            // 但如果原先就是1 的情况 无法处理 就需要 特殊处理成 0
            if (I == OptPutIndex) {
                for (size_t dim_i = 0; dim_i < operands_[I].strides.size(); ++dim_i) {
                    if (is_reduce_dim(static_cast<index_t>(dim_i))) {
                        operands_[I].strides[dim_i] = 0;
                    }
                }
            }
        }

        template <std::size_t I>
        void calc_strides_for_operand() {
            operands_[I].strides_bytes = calc_strides_bytes(operands_[I].ScalarType, operands_[I].strides);
        }

        void compute_output_shape() {
            if (!shape_.empty() && get_tensor<OptPutIndex>().get_shape().dims() != shape_) {
                auto output_dims = std::vector(shape_.begin(), shape_.end());
                if (config_.is_reduction_) {
                    //遍历要 reduce的维度 然后 reduce过后形状设置为 1
                    for (const auto reduce_dim: config_.reduce_dims_) {
                        output_dims[reduce_dim] = 1;
                    }
                }
                auto new_shape = Shape(output_dims);
                get_tensor<OptPutIndex>().set_initial_shape(new_shape);
                operands_[OptPutIndex] = OperandInfo(const_cast<ResultScalarType*>(
                    get_tensor<OptPutIndex>().mutable_data() + get_tensor<OptPutIndex>().get_offset()),
                    get_tensor<OptPutIndex>().get_strides(), true, true);
            }

        }

        void init_default() {
            if (config_.is_reduction_) reduce_dim_size_ = config_.reduce_dims_.size();
        }

        void analysis_memory_layout() {
            is_contiguous_ = true;
            // 如果 所有操作的输入张量都是连续的 就标记连续
            for (const auto& op : operands_) {
                is_contiguous_ &= Utils::is_contiguous(op.strides, shape_);
            }
            is_reduce_dim_contiguous_ =  reduce_dim_size_ > 0 && operands_[0].strides[reduce_dim_size_ - 1] == 1;
            // 针对最后一个维度连续 可以对连续的维度 拆分n次循环外部循环 和 内部连续的循环
            is_inner_contiguous_ = true;
            for (const auto& op : operands_) {
                if (!op.strides.empty() && op.strides[0] != 1) {
                    is_inner_contiguous_ = false;
                    break;
                }
            }
        }


        void resize_output() {
            if (!(!config_.keep_keepdim_ && config_.is_reduction_))
                return;

            const size_t original_reduce_size = config_.reduce_dims_.size();
            const size_t input_dims = shape_.size();
            const size_t output_dims = input_dims - reduce_dim_size_;

            BREEZE_ASSERT(input_dims >= reduce_dim_size_,
                          "Invalid reduction: input dimensions (", input_dims,
                          ") must be greater than or equal to reduce dimensions (", reduce_dim_size_, ")");

            auto final_shape = std::vector<index_t>();
            auto final_strides = std::vector<index_t>();
            final_shape.reserve(output_dims);
            final_strides.reserve(output_dims);

            std::vector<index_t> original_shape = get_shape_from_tuple<0>();
            std::vector<index_t> original_stides = get_tensor<0>().get_strides();

            // 根据 perm_ 重新排列 final_shape 和 final_strides
            for (size_t i = 0; i < original_shape.size(); ++i) {
                if (auto it =
                    std::find(perm_.begin() + original_reduce_size, perm_.end(), i); it!= perm_.end()) {
                    final_shape.push_back(original_shape[i]);
                    final_strides.push_back(get_tensor<OptPutIndex>().get_strides()[i]);
                }
            }

            // 检查最终形状是否有效
            BREEZE_ASSERT(std::all_of(final_shape.begin(), final_shape.end(), [](const index_t dim) { return dim > 0; }),
                          "Invalid final shape: all dimensions must be positive");

            // 更新输出张量的形状和步长
            get_tensor<OptPutIndex>().set_shape_and_strides(final_shape, final_strides, true);
        }

        template <std::size_t... Indices>
        void build(std::index_sequence<Indices...>) {

            init_default();

            if (config_.check_all_same_dtype_) {
                check_all_same_dtype();
            }

            if (config_.check_all_same_shape_) {
                (check_all_same_shape<Indices>(), ...);
            }

            shape_ = compute_common_shape(std::make_index_sequence<sizeof...(ScalarTypes)>{});

            compute_output_shape();

            (expand_strides_for_operand<Indices>(), ...);

            reorder_dimensions();

            coalesce_dimensions(std::make_index_sequence<sizeof...(ScalarTypes)>{});

            (calc_strides_for_operand<Indices>(), ...);

            if (config_.enforce_safe_casting_to_output_) {
                check_safe_to_output(std::make_index_sequence<sizeof...(ScalarTypes)>{});
            }

            if (config_.resize_outputs_) {
                resize_output();
            }

            analysis_memory_layout();

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

        template<typename ReduceScalarOp, typename ResultScalarType, typename ReduceVectorOp, size_t... Indices>
        void reduce_non_contiguous(const char* data_ptr, const std::vector<index_t>& reduce_shape,
                                   const std::vector<index_t>& strides_bytes, ReduceScalarOp reduce_scalar_op,
                                   ReduceVectorOp reduce_vector_op, const index_t reduce_size, std::array<ResultScalarType, SimdVrctorSize> &local_reduce_vec,
                                   const index_t simd_unaligned_start_offset, std::index_sequence<Indices...>) {
            std::vector<index_t> reduce_counter(reduce_shape.size());
            std::array<const char*, SimdVrctorSize> vec_data_ptrs;

            for (index_t j = 0; j  <= reduce_size - SimdVrctorSize; j += SimdVrctorSize) {
                // Calculate offsets for SimdVectorSize elements at once
                for (index_t i = 0; i < SimdVrctorSize && (j + i) < reduce_size; ++i) {
                    Utils::index_to_counter(j + i, reduce_counter, reduce_shape);
                    vec_data_ptrs[i] = data_ptr + Utils::compute_offset(reduce_counter, strides_bytes, 0);
                }
                // Load and reduce vector data
                reduce_vector_op(reinterpret_cast<ResultScalarType*>(&local_reduce_vec),
                                 Vectorized<ResultScalarType>::loadu_unaligned(
                                     reinterpret_cast<const ResultScalarType*>(vec_data_ptrs[Indices])...));
            }

            // Handle remaining elements that don't fit in a full SIMD vector
            for (index_t j = simd_unaligned_start_offset; j < reduce_size; ++j) {
                Utils::index_to_counter(j, reduce_counter, reduce_shape);
                const char* scalar_ptr = data_ptr + Utils::compute_offset(reduce_counter, strides_bytes, 0);
                reduce_scalar_op(reinterpret_cast<ResultScalarType*>(&local_reduce_vec) +
                    (j % SimdVrctorSize) , *reinterpret_cast<const ResultScalarType*>(scalar_ptr));
            }
        }

        template<typename InitValueFunc, typename ReduceScalarOp, typename ReduceVectorOp, typename FinalReduceOp>
        void reduce_strided_for_each(InitValueFunc init_value_func, ReduceScalarOp reduce_scalar_op, ReduceVectorOp reduce_vector_op,
            FinalReduceOp final_reduce_op) {
            std::vector<index_t> non_reduce_shape(shape_.size() - reduce_dim_size_);
            std::vector<index_t> reduce_shape;
            index_t non_reduce_size = 1;
            index_t reduce_size = 1;
            for (size_t i = 0; i < non_reduce_shape.size(); ++i) {
                non_reduce_shape[i] = shape_[i + reduce_dim_size_];
                non_reduce_size *= non_reduce_shape[i];
            }

            for (size_t j = 0; j < reduce_dim_size_; ++j) {
                reduce_shape.push_back(shape_[j]);
                reduce_size *= reduce_shape[j];
            }

            const index_t simd_unaligned_start_offset = reduce_size - reduce_size % SimdVrctorSize;
            const index_t final_reduce_size = reduce_size < SimdVrctorSize ? reduce_size : SimdVrctorSize;

            #pragma omp parallel default(none) shared(reduce_size, reduce_shape, \
                non_reduce_shape, non_reduce_size, reduce_vector_op, reduce_scalar_op, \
                final_reduce_op, init_value_func,operands_, is_reduce_dim_contiguous_, final_reduce_size, simd_unaligned_start_offset)
            {
                alignas(64) std::array<char*, ScalarTypesSize> tile_data_ptrs;
                std::array<ResultScalarType, SimdVrctorSize> local_reduce_vec = {};
                std::vector<index_t> non_reduce_counter(non_reduce_shape.size());

                auto process_tile = [&](const index_t i) {
                    init_value_func(reinterpret_cast<ResultScalarType*>(&local_reduce_vec));
                    Utils::index_to_counter(i, non_reduce_counter, non_reduce_shape);
                    for (size_t k = 0; k < operands_.size(); ++k) {
                        tile_data_ptrs[k] = operands_[k].data + Utils::compute_offset(non_reduce_counter, operands_[k].strides_bytes);
                    }

                    if (is_reduce_dim_contiguous_) {
                        for (index_t j = 0; j <= reduce_size - SimdVrctorSize; j += SimdVrctorSize) {
                            reduce_vector_op(reinterpret_cast<ResultScalarType*>(&local_reduce_vec),
                                Vectorized<ResultScalarType>::loadu(reinterpret_cast<ResultScalarType*>(tile_data_ptrs[0]) + j));
                        }
                        for (index_t j = simd_unaligned_start_offset; j < reduce_size; ++j) {
                            const index_t vec_offset = j % SimdVrctorSize;
                            reduce_scalar_op(reinterpret_cast<ResultScalarType*>(&local_reduce_vec) + vec_offset,
                                *(reinterpret_cast<ResultScalarType*>(tile_data_ptrs[0]) + j));
                        }
                    } else {
                        reduce_non_contiguous(tile_data_ptrs[0], reduce_shape,
                            operands_[0].strides_bytes, reduce_scalar_op, reduce_vector_op,
                            reduce_size, local_reduce_vec,
                            simd_unaligned_start_offset, std::make_index_sequence<SimdVrctorSize>{});
                    }

                    ResultScalarType tile_result = final_reduce_op(local_reduce_vec.data(), final_reduce_size);
                    *reinterpret_cast<ResultScalarType*>(tile_data_ptrs[OptPutIndex]) = tile_result;
                };

                #pragma omp for schedule(dynamic, 1) nowait
                for (index_t tile_start = 0; tile_start < non_reduce_size; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, non_reduce_size);
                    for (index_t i = tile_start; i < tile_end; ++i) {
                        process_tile(i);
                    }
                }
            }
        }


        template<typename ForEachFunc>
        void contiguous_for_each(ForEachFunc for_each_func) {
            BREEZE_ASSERT(shape_.size() == 1,
                "Invalid shape for contiguous_for_each: expected contiguous tensor");
            const index_t numel = shape_.back();
            #pragma omp parallel default(none) \
            shared(operands_, for_each_func) firstprivate(numel)
            {
                alignas(64) std::array<char*, ScalarTypesSize> tile_data_ptrs;
                std::array<index_t, ScalarTypesSize> tile_data_strides;
                #pragma omp for schedule(dynamic, 64)
                for (index_t tile_start = 0; tile_start < numel; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, numel);
                    for (size_t k = 0; k < ScalarTypesSize; ++k) {
                        BREEZE_ASSERT(operands_[k].strides_bytes.size() == 1,  "Invalid strides for operand ", k,
                            ": expected size 1, but got ", operands_[k].strides_bytes.size(), ". Operands must have 1-dimensional strides.");
                        tile_data_ptrs[k] = operands_[k].data + tile_start * operands_[k].strides_bytes[0];
                        tile_data_strides[k] = operands_[k].strides_bytes.back();
                    }
                    for_each_func(tile_data_ptrs.data(), tile_data_strides.data(), tile_end - tile_start);
                }
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void contiguous_kernel_vec(ScalarOp scalar_op, VectorOp vector_op) {
            BREEZE_ASSERT(shape_.size() == 1,
              "Invalid shape for contiguous_for_each: expected contiguous tensor");
            const index_t numel = shape_.back();

            #pragma omp parallel default(none) \
            shared(operands_, scalar_op, vector_op) firstprivate(numel, shape_)
            {
                alignas(64) std::array<char*, ScalarTypesSize> tile_data_ptrs;

                #pragma omp for schedule(static)
                for (index_t tile_start = 0; tile_start < numel; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, numel);
                    const index_t tile_elements = tile_end - tile_start;

                    for (size_t k = 0; k < ScalarTypesSize; ++k) {
                        BREEZE_ASSERT(operands_[k].strides_bytes.size() == 1 &&
                                      operands_[k].strides_bytes[0] == ResultTypeSize,
                            "Invalid strides for operand " + std::to_string(k));
                        tile_data_ptrs[k] = operands_[k].data + tile_start * ResultTypeSize;
                    }
                    alignas(64) char* tile_output_ptr = tile_data_ptrs[OptPutIndex];
                    op_loop(scalar_op, vector_op, tile_data_ptrs, tile_output_ptr, tile_elements);
                }
            }
        }


        template<typename ForEachFunc>
        void inner_contiguous_for_each(ForEachFunc for_each_func) {
            const index_t inner_dim = shape_.back();
            const index_t outer_dim = std::accumulate(shape_.begin() + 1,
                shape_.end(), 1LL, std::multiplies<>());

            #pragma omp parallel default(none) \
            shared(operands_, for_each_func) firstprivate(inner_dim, outer_dim, shape_)
            {
                alignas(64) std::array<char*, ScalarTypesSize> tile_data_ptrs;
                std::array<index_t, ScalarTypesSize> tile_data_strides;
                std::vector<index_t> tile_counter(shape_.size() - 1);

                #pragma omp for schedule(static)
                for (index_t tile_start = 0; tile_start < outer_dim; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, outer_dim);
                    Utils::index_to_counter(tile_start, tile_counter, shape_);

                    for (size_t k = 0; k < ScalarTypesSize; ++k) {
                        BREEZE_ASSERT(operands_[k].strides[0] == 1, "Invalid strides for operand " + std::to_string(k));
                        tile_data_ptrs[k] = operands_[k].data + Utils::compute_offset(tile_counter, operands_[k].strides_bytes);
                        tile_data_strides[k] = operands_[k].strides[0];
                    }

                    for (index_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                        for_each_func(tile_data_ptrs.data(), tile_data_strides.data(), inner_dim);
                    }
                }
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void inner_contiguous_kernel_vec(ScalarOp scalar_op, VectorOp vector_op) {
            const index_t inner_size = shape_.back();
            const index_t outer_size = std::accumulate(shape_.begin() + 1, shape_.end(), 1LL, std::multiplies<>());

            #pragma omp parallel default(none) \
            shared(operands_, scalar_op, vector_op) firstprivate(outer_size, inner_size, shape_)
            {
                alignas(64) std::array<char*, ScalarTypesSize> tile_data_ptrs;
                std::vector<index_t> tile_counter(shape_.size() - 1);

                #pragma omp for schedule(static)
                for (index_t tile_start = 0; tile_start < outer_size; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, outer_size);
                    for (index_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                        Utils::index_to_counter(tile_idx, tile_counter, shape_);

                        for (size_t k = 0; k < ScalarTypesSize; ++k) {
                            BREEZE_ASSERT(operands_[k].strides[0] == 1,  "Invalid strides for operand ", k,
                            ": expected last stride 1, but got ", operands_[k].strides[0], ". Operands last stride must contiguous.");
                            tile_data_ptrs[k] = operands_[k].data + Utils::compute_offset(tile_counter, operands_[k].strides_bytes);

                        }
                        alignas(64) char* output_ptr = tile_data_ptrs[OptPutIndex];
                        op_loop(scalar_op, vector_op, tile_data_ptrs, output_ptr, inner_size);
                    }
                }
            }
        }

        template<typename ForEachFunc>
        void strided_for_each(ForEachFunc for_each_func) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(),
                1LL, std::multiplies<>());

            #pragma omp parallel default(none) \
            shared(for_each_func, operands_) firstprivate(numel, shape_)
            {
                alignas(64)  std::array<char*, ScalarTypesSize> tile_data_ptrs;
                std::array<index_t, ScalarTypesSize> tile_data_strides;
                std::vector<index_t> tile_counter(shape_.size());

                #pragma omp for schedule(static)
                for (index_t tile_start = 0; tile_start < numel; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, numel);
                    Utils::index_to_counter(tile_start, tile_counter, shape_);

                    for (index_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                        for (size_t k = 0; k < ScalarTypesSize; ++k) {
                            tile_data_ptrs[k] = operands_[k].data + Utils::compute_offset(tile_counter, operands_[k].strides_bytes, 0);
                            tile_data_strides[k] = 1;
                        }
                        for_each_func(tile_data_ptrs.data(), tile_data_strides.data(), 1);
                    }
                }
            }
        }


        template<typename ScalarOp, typename VectorOp>
        void strided_kernel_vec(ScalarOp scalar_op, VectorOp vector_op) {
            const index_t numel = std::accumulate(shape_.begin(), shape_.end(),
                1LL, std::multiplies<>());

            #pragma omp parallel default(none) \
            shared(scalar_op, vector_op, operands_) firstprivate(numel, shape_)
            {
                alignas(64) std::array<std::array<char*, SimdVrctorSize>, ScalarTypesSize> tile_data_ptrs;
                std::vector<index_t> tile_counter(shape_.size());

                #pragma omp for schedule(static)
                for (index_t tile_start = 0; tile_start < numel; tile_start += TILE_SIZE) {
                    const index_t tile_end = std::min(tile_start + TILE_SIZE, numel);
                    Utils::index_to_counter(tile_start, tile_counter, shape_);

                    for (index_t tile_idx = tile_start; tile_idx < tile_end; tile_idx += SimdVrctorSize) {
                        const index_t remaining = std::min(SimdVrctorSize, tile_end - tile_idx);

                        for (int i = 0; i < remaining; ++i) {
                            for (size_t k = 0; k < ScalarTypesSize; ++k) {
                                tile_data_ptrs[k][i] = operands_[k].data +
                                    Utils::compute_offset(tile_counter, operands_[k].strides_bytes, 0);
                            }
                            Utils::increment_counter(tile_counter, shape_, 1);
                        }

                        if (remaining == SimdVrctorSize) {
                            simd_loop_impl(vector_op, tile_data_ptrs, tile_data_ptrs[OptPutIndex], std::make_index_sequence<ScalarTypesSize - 1>{});
                        } else {
                            scalar_loop_impl(scalar_op, tile_data_ptrs, tile_data_ptrs[OptPutIndex], remaining, std::make_index_sequence<ScalarTypesSize - 1>{});
                        }
                    }
                }
            }
        }

        template<typename VectorOp, size_t... Indices>
        static void simd_loop_impl(VectorOp vector_op,
                           const std::array<std::array<char*, SimdVrctorSize>, ScalarTypesSize>& data_ptrs,
                           const std::array<char*, SimdVrctorSize>& output_ptrs,
                           std::index_sequence<Indices...>) {
            alignas(64) std::array<ResultScalarType, SimdVrctorSize> temp_output;
            auto load_simd = [](const std::array<char*, SimdVrctorSize>& ptrs) {
                alignas(64) ResultScalarType temp[SimdVrctorSize];
                for (int i = 0; i < SimdVrctorSize; ++i) {
                    temp[i] = *reinterpret_cast<const ResultScalarType*>(ptrs[i]);
                }
                return Vectorized<ResultScalarType>::loadu(temp);
            };
            std::array<Vectorized<ResultScalarType>, sizeof...(Indices)> input_vecs = {
                load_simd(data_ptrs[Indices])...
            };
            // 调用vector_op
            vector_op(temp_output.data(), input_vecs[Indices]...);
            // 存储结果
            for (size_t i = 0; i < SimdVrctorSize; ++i) {
                *reinterpret_cast<ResultScalarType*>(output_ptrs[i]) = temp_output[i];
            }
        }

        template<typename ScalarOp, size_t... Indices>
        static void scalar_loop_impl(ScalarOp scalar_op,
                             const std::array<std::array<char*, SimdVrctorSize>, ScalarTypesSize>& data_ptrs,
                             const std::array<char*, SimdVrctorSize>& output_ptrs,
                             const index_t size,
                             std::index_sequence<Indices...>) {
            for (index_t i = 0; i < size; ++i) {
                scalar_op(reinterpret_cast<ResultScalarType*>(output_ptrs[i]),
                          *reinterpret_cast<const ResultScalarType*>(data_ptrs[Indices][i])...);
            }
        }

    private:
        std::array<OperandInfo, sizeof ...(ScalarTypes)> operands_ ;
        std::tuple<Tensor<ScalarTypes>*...> tensors_;
        std::vector<index_t> shape_{};
        std::vector<index_t> perm_{};
        size_t reduce_dim_size_ = 0;
        bool is_contiguous_= false;
        bool is_inner_contiguous_= false;
        bool is_reduce_dim_contiguous_= false;
        bool has_coalesced_dimensions_= false;
        TensorIteratorConfig config_{};

        template<size_t I>
        auto& get_tensor() const{
            return *std::get<I>(tensors_);
        }

        template<size_t I>
        auto& get_tensor(){
            return *std::get<I>(tensors_);
        }

        template<size_t I>
        [[nodiscard]] index_t get_tensor_data_size() const{
            return get_tensor<I>().size();
        }

        // shape[n] * stride[n] == stride[n + 1]. 或者 维度为 1 的 可以合并
        template<size_t... Indices>
        [[nodiscard]] bool can_coalesce(const index_t dim0, const index_t dim1,
            std::index_sequence<Indices...>) const {

            // 排过序后 前面的事reduce的维度 后面是普通维度 相同维度可以归约
            if (const auto reduce_ele_nums = static_cast<index_t>(config_.reduce_dims_.size());
                dim1 - 1 <  reduce_ele_nums != dim1 < reduce_ele_nums) {
                return false;
            }

            if (shape_[dim0] == 1 || shape_[dim1] == 1) {
                return true;
            }

            auto check_contiguous = [this, dim0, dim1](const auto& op) {
                const auto& strides = op.strides;
                return shape_[dim0] * strides[dim0] == strides[dim1];
            };
            if (!(check_contiguous(get_operand<Indices>()) && ...)) {
                return false;
            }
            return true;
        }

        template<size_t... Indices>
        void coalesce_dimensions(std::index_sequence<Indices...>) {
            if (shape_.size() <= 1) return;
            auto replace_strides = [&](int dim0, int dim1) {
                ((get_operand<Indices>().strides[dim0] = get_operand<Indices>().strides[dim1]), ...);
            };
            index_t last_dim = 0;
            for (index_t dim_i = 1; dim_i < ndim(); ++dim_i) {
                // 连续的情况 得规约 比较所有 要操作的张量的步长 如果每个操作张量都是连续的 则进行规约
                // 比如 shape(5 4 3 4) 步长 (12 0 4 1) => dim_i = 1 shape[1] = 4 strides[1] = 0
                // 聚合维度不参与 为了后面计算 需要保留信息
                if (can_coalesce(last_dim, dim_i, std::make_index_sequence<sizeof... (Indices)>{})) {
                    if (dim_i < static_cast<index_t>(config_.reduce_dims_.size())) {
                        --reduce_dim_size_;
                    }
                    // 可以合并的情况
                    // 情况1：如果前一个维度的 shape 为 1
                    // 例如：shape(1, 4, 3, 4) -> shape(4, 3, 4)
                    // 步长：(12, 3, 1) -> (3, 1)
                    // 1 * n = n，可以直接替换步长，减少后续不必要的乘法操作
                    if (shape_[last_dim] == 1) {
                        replace_strides(last_dim, dim_i);
                    }
                    // 情况2：步长连续，则合并形状
                    // 例如：shape(5, 4, 3, 4) -> shape(20, 3, 4)
                    // 步长：(48, 12, 4, 1) -> (12, 4, 1)
                    shape_[last_dim] *= shape_[dim_i];
                    // 注意：这里不需要更新步长，因为合并后的步长保持不变
                    // 例如：(48, 12, 4, 1) 合并后仍然是 (12, 4, 1)
                }else {
                    // 不能合并的情况
                    // 例如，考虑形状和步长：shape(5, 4, 3, 4), strides(12, 0, 4, 1)
                    // 这里，第二个维度（4）是被广播的，因此其步长为0
                    //
                    // 在处理这种情况时：
                    // 1. 我们不能合并第二个维度（步长为0）和第三个维度（步长为4）
                    // 2. 但我们需要在逻辑表示中保留这个广播维度
                    //
                    // 结果可能保持为：shape(5, 4, 3, 4), strides(12, 0, 4, 1)
                    //
                    // 这种处理确保了：
                    // - 广播维度被正确地保留在逻辑表示中
                    // - 后续的计算操作可以正确地处理这个广播维度
                    // - 内存访问模式反映了实际的数据布局，包括广播维度
                    // 移动到下一个需要处理的维度
                    ++last_dim;
                    // 如果当前维度与新的 last_dim 不同，需要更新形状和步长
                    if (last_dim != dim_i) {
                        // 更新步长，将 dim_i 的步长复制到 last_dim
                        replace_strides(last_dim, dim_i);
                        // 更新形状
                        shape_[last_dim] = shape_[dim_i];
                    }
                    // 如果 last_dim == dim_i，则不需要任何操作，因为它们已经在正确的位置
                }
            }
            shape_.resize(last_dim + 1);
            (operands_[Indices].strides.resize(ndim()),...);
            has_coalesced_dimensions_ = true;
        }

        void reorder_dimensions() {
            perm_ = std::vector<index_t>(ndim(),0);
            //维度<= 1 返回
            if (ndim() <= 1)
                return;
            // 初始化 perm with n-1, n-2, ..., 1, 0
            std::iota(perm_.rbegin(), perm_.rend(), 0);

            // 重排strides 从末尾 n - 1 n - 2 n - 3  ...
            if (config_.enforce_linear_iteration_) {
                permute_dimensions(perm_);
                return;
            }

            // 按照步长的大小排序 这里从最后一个遍历 因为最后一个如果 设置了输出大小 如果reduce了某个维度应该 优先移到最前
            auto need_swap = [&](size_t dim0, size_t dim1) -> int {
                for (index_t i = operands_.size() -1; i >= 0; --i) {
                    const auto& op = operands_[i];
                    // 忽略输出需要计算大小的 tensor
                    if (op.strides.empty()) {
                        continue;
                    }
                    const int64_t stride0 = op.strides[dim0];
                    const int64_t stride1 = op.strides[dim1];
                    if (config_.is_reduction_ && op.is_output) {
                        if ((stride1 == 0) != (stride0 == 0)) {
                            return stride1 == 0 ? 1 : -1; // 等价于 (stride1 == 0) - (stride0 == 0) 0步长总是在最前面
                        }
                    }
                    // 如果 stride为 0 表示不连续 一般是复制的维度所以依赖于后面的维度用来复制 所以不能重排
                    if (stride1 == 0 || stride0 == 0) {
                        continue;
                    }
                    if (stride1 != stride0) { // 如果前一个维度大于后面的维度 返回1
                        return stride0 > stride1 ? 1 : -1;
                    }
                    // 如果不满足则进入下一个循环
                    if (shape_[dim0] > shape_[dim1]) return 1;
                }
                return 0;
            };

            //插入排序 对步长从小到大排序
            for (index_t i = 1; i < ndim(); ++i) {
                index_t dim1 = i;
                for (index_t dim0 = i - 1; dim0 >= 0; --dim0) {
                    if (const index_t comparison = need_swap(perm_[dim0], perm_[dim1]); comparison > 0) {
                        std::swap(perm_[dim0], perm_[dim1]);
                        dim1 = dim0;
                    } else if (comparison < 0) {
                        break;
                    }
                }
            }
            // 计算出 perm_ 后对 strides 和 shape_重排
            permute_dimensions(perm_);
        }

        template<typename VectorOp, size_t... Indices>
        static void vectorized_loop_impl(VectorOp vector_op, const std::array<char*, ScalarTypesSize>& data_ptrs,
                              char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<Indices...>) {

            index_t prefetch_counter = -1;
            for (index_t i = 0; i <= size - simd_vector_size; i += simd_vector_size) {
                // 执行向量化操作 如果要实现不同类型操作可以在 loadu 添加转换
                vector_op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                   Vectorized<ResultScalarType>::loadu(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i)...);
                // 每处理 prefetch_elements 个向量后进行一次预读取
                if (++prefetch_counter == PrefetchElements) {
                    prefetch_counter = -1;
                    if (i < size - PrefetchElements) {
                        Vectorized<ResultScalarType>::prefetch(reinterpret_cast<ResultScalarType*>(output_ptr) + i + PrefetchElements);
                        (Vectorized<ResultScalarType>::prefetch(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i + PrefetchElements), ...);
                    }
                }
            }
        }


        template<typename ScalarOp, size_t... Indices>
            static void scalar_loop_impl(ScalarOp scalar_op, const std::array<char*, ScalarTypesSize> &data_ptrs,
                char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<Indices...>) {
            const auto begin = size - size % simd_vector_size;
            for (index_t i = begin; i < size; i += 1) {
                //这里 不同类型转换使用默认编译器规则
                scalar_op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                   *(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i)...);
            }
        }

        template<typename ScalarOp, typename VectorOp>
        void op_loop(ScalarOp scalar_op, VectorOp vector_op,
            const std::array<char*, ScalarTypesSize> &data_ptrs, char* output_ptr, const index_t size) {
            vectorized_loop_impl(vector_op, data_ptrs, output_ptr, size, SimdVrctorSize, std::make_index_sequence<ScalarTypesSize - 1>{});
            scalar_loop_impl(scalar_op, data_ptrs, output_ptr, size, SimdVrctorSize, std::make_index_sequence<ScalarTypesSize - 1>{});
        }

        void check_all_same_dtype() const{
            ScalarType first_type = operands_[0].ScalarType;
            for (size_t i = 1; i < operands_.size(); ++i) {
                BREEZE_ASSERT(operands_[i].ScalarType == first_type,
                              "Data type mismatch: Operand at index ", i, " has type ",
                              scalar_type_to_string(operands_[i].ScalarType),
                              " which differs from the first operand's type ",
                              scalar_type_to_string(first_type),
                              ". All tensors must have the same dtype.");
            }
        }

        template<size_t... Indices>
        void check_safe_to_output(std::index_sequence<Indices...>) const {
            constexpr size_t num_tensors = sizeof...(Indices);
            std::array<index_t, num_tensors> data_size = { get_tensor_data_size<Indices>()... };
            const index_t output_size = data_size[num_tensors - 1];

            // Check each input tensor against the output tensor
            bool is_safe = true;
            ((is_safe &= (Indices == num_tensors - 1 || output_size >= data_size[Indices])), ...);

            BREEZE_ASSERT(is_safe,
                          "Unsafe output operation: Output tensor (size ", output_size,
                          ") does not have enough memory to store the result. ",
                          "Input tensor sizes: ", [&]() {
                              std::string sizes;
                              for (size_t i = 0; i < num_tensors - 1; ++i) {
                                  sizes += std::to_string(data_size[i]) + (i < num_tensors - 2 ? ", " : "");
                              }
                              return sizes;
                          }());
        }

        template<std::size_t I>
        void check_all_same_shape() const {
            if (I == 0) return;
            const auto first_shape = get_shape_from_tuple<0>();
            // BREEZE_ASSERT(get_shape_from_tuple<I>() == first_shape,
            //               "Shape mismatch: tensor at index ", I, " has shape ",
            //               get_shape_from_tuple<I>(), " which differs from the first tensor's shape ",
            //               first_shape, ". All tensors must have the same shape when no_broadcast is set.");
        }

        [[nodiscard]] bool is_reduce_dim(const index_t dim) const{
            if (!config_.is_reduction_)
                return false;
            const auto it = std::find(config_.reduce_dims_.begin(), config_.reduce_dims_.end(), dim);
            return it != config_.reduce_dims_.end();
        }

        [[nodiscard]] std::vector<index_t> invert_perm(const std::vector<index_t>& input) const {
            BREEZE_ASSERT(has_coalesced_dimensions_,
                          "Invalid state: dimensions are not coalesced");
            BREEZE_ASSERT(input.size() == perm_.size(),
                          "Invalid input: size mismatch between input (", input.size(),
                          ") and perm_ (", perm_.size(), ")");
            auto res = std::vector<index_t>(input.size());
            for (index_t dim = 0; dim < ndim(); ++dim) {
                res[perm_[dim]] = input[dim];
            }
            return res;
        }

        // 辅助函数：获取特定索引的操作数
        template<size_t I>
        const auto& get_operand() const {
            return std::get<I>(operands_);
        }

        template<size_t I>
        auto& get_operand() {
            return std::get<I>(operands_);
        }

        template<std::size_t... Indices>
        std::vector<index_t> compute_common_shape(std::index_sequence<Indices...>) {
            if constexpr (sizeof...(Indices) == 1) {
                return std::vector<index_t>(get_tensor<OptPutIndex>().get_shape().dims().begin(),
                    get_tensor<OptPutIndex>().get_shape().dims().end());;
            }else if constexpr (sizeof...(Indices) >= 2) {
                return Utils::broadcast_shapes(get_shape_from_tuple<0>(), get_shape_from_tuple<1>());
            }
            T_ERROR("none tensor to broadcast");
        }

        void permute_dimensions(std::vector<index_t>& perm) {
            BREEZE_ASSERT(perm.size() == static_cast<size_t>(ndim()),
                          "Invalid permutation: size mismatch. Expected ", ndim(),
                          " dimensions, but got ", perm.size(), " in the permutation.");

            // Check if the permutation is valid (contains all indices exactly once)
            std::vector<bool> used(ndim(), false);
            for (const auto& idx : perm) {
                BREEZE_ASSERT(idx >= 0 && idx < ndim(),
                              "Invalid permutation index: ", idx,
                              ". Must be between 0 and ", ndim() - 1, " (inclusive).");
                BREEZE_ASSERT(!used[idx],
                              "Invalid permutation: index ", idx, " appears more than once.");
                used[idx] = true;
            }

            auto reorder = [&](const auto& vec) {
                auto res = vec;
                for (size_t i = 0; i < perm.size(); i++) {
                    res[i] = vec[perm[i]];
                }
                return res;
            };

            shape_ = reorder(shape_);
            for (auto& op : operands_) {
                if (!op.strides.empty()) {
                    op.strides = reorder(op.strides);
                }
            }
        }

        [[nodiscard]] index_t ndim() const {
            return shape_.size();
        }

    };
}

#endif //TENSORITERATOR_H