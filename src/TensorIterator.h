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
            operands_[I].strides = Utils::expand_strides(get_shape_from_tuple<I>(), shape_, operands_[I].strides);
        }

        template <std::size_t I>
        void calc_strides_for_operand() {
            operands_[I].strides_bytes = calc_strides_bytes(operands_[I].ScalarType, operands_[I].strides);
        }

        template <std::size_t... Indices>
        void build(std::index_sequence<Indices...>) {

            if (config_.check_all_same_dtype_) {
                check_all_same_dtype();
            }

            if (config_.check_all_same_shape_) {
                (check_all_sanme_shape<Indices>(), ...);
            }

            if (ScalarTypesSize > 1) shape_ = compute_common_shape(std::make_index_sequence<2>{});

            if (shape_.empty()) {
                shape_ = std::vector<index_t>(get_tensor<OptPutIndex>().get_shape().dims().begin(),
                get_tensor<OptPutIndex>().get_shape().dims().end());
            }

            if (config_.resize_outputs_) {
                if (!shape_.empty() && get_tensor<OptPutIndex>().get_shape().dims() != shape_) {
                    auto new_shape = Shape(std::vector(shape_.begin(), shape_.end()));
                    get_tensor<OptPutIndex>().set_initial_shape(new_shape);
                    operands_[OptPutIndex] = OperandInfo(const_cast<OutputScalarType*>(get_tensor<OptPutIndex>().mutable_data()),
                        get_tensor<OptPutIndex>().get_strides(),
                        get_tensor<OptPutIndex>().get_offset(), true, true);
                }
            }

            (expand_strides_for_operand<Indices>(), ...);

            reorder_dimensions();

            coalesce_dimensions(std::make_index_sequence<sizeof...(ScalarTypes)>{});

            (calc_strides_for_operand<Indices>(), ...);

            if (config_.enforce_safe_casting_to_output_) {
                check_safe_to_output(std::make_index_sequence<sizeof...(ScalarTypes)>{});
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
        std::vector<index_t> perm_{};
        bool is_contiguous_{};
        bool is_inner_contiguous_{};
        size_t common_dim_{};
        TensorIteratorConfig config_{};

        template<size_t I>
        auto& get_tensor() const{
            return *std::get<I>(tensors_);
        }

        template<size_t I>
        [[nodiscard]] index_t get_tensor_data_size() const{
            return get_tensor<I>().size();
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
            const index_t numel = shape_[0];
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
                    local_ptrs[j] += Utils::compute_offset(counter, operands_[j].strides_bytes);
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
                            Utils::compute_offset(counter, operands_[k].strides_bytes);
                    }
                    char* output_ptr = data_ptrs[OptPutIndex];
                    scalar_loop_impl(scalar_op, data_ptrs, output_ptr, 1, 1, std::make_index_sequence<ScalarTypesSize - 1>{});
                    increment_counter(counter);
                }
            }
        }

        // shape[n] * stride[n] == stride[n + 1]. 或者 维度为 1 的 可以合并
        template<size_t... Indices>
        [[nodiscard]] bool can_coalesce(const index_t dim0, const index_t dim1,
            std::index_sequence<Indices...>) const {

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
            for (size_t dim_i = 1; dim_i < ndim(); ++dim_i) {
                // 连续的情况 得规约 比较所有 要操作的张量的步长 如果每个操作张量都是连续的 则进行规约
                // 比如 shape(5 4 3 4) 步长 (12 0 4 1) => dim_i = 1 shape[1] = 4 strides[1] = 0
                if (can_coalesce(last_dim, dim_i, std::make_index_sequence<sizeof... (Indices) -1>{})) {
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

            // 按照步长的大小排序
            auto need_swap = [&](size_t dim0, size_t dim1) -> int {
                for (index_t i = 0; i < operands_.size(); ++i) {
                    const auto& op = operands_[i];
                    // 忽略输出需要计算大小的 tensor
                    if (op.strides.empty() || (op.is_output && config_.resize_outputs_)) {
                        continue;
                    }
                    const int64_t stride0 = op.strides[dim0];
                    const int64_t stride1 = op.strides[dim1];
                    if (config_.is_reduction_ && op.is_output) {
                        if ((stride1 == 0) != (stride0 == 0)) {
                            return stride0 == 0 ? 1 : -1;// 等价于 (stride1 == 0) - (stride0 == 0) 0步长总是在最前面
                        }
                    }
                    if (stride1 == 0 || stride0 == 0) {
                        continue;
                    }
                    if (stride1 != stride0) {
                        return stride1 > stride0 ? 1 : -1;
                    }
                }
                // 如果所有操作数的stride都相等，使用shape作为tie-breaker
                return shape_[dim1] > shape_[dim0] ? 1 :  -1 ;
            };

            //插入排序 对步长从大到小排序
            for (index_t i = 1; i < ndim(); ++i) {
                index_t dim0 = i;
                for (index_t dim1 = i - 1; dim1 >= 0; --dim1) {
                    if (const index_t comparison = need_swap(perm_[dim0], perm_[dim1]); comparison > 0) {
                        std::swap(perm_[dim1], perm_[dim0]);
                        dim0 = dim1;
                    } else if (comparison < 0) {
                        break;
                    }
                }
            }
            // 计算出 perm_ 后对 strides 重排
            permute_dimensions(perm_);
        }


        template<typename Func>
        void reduce_strided_for_each(Func& reduce_func) {

            // std::vector<index_t> non_reduce_counter(shape_.size() -1, 0);
            // std::vector<index_t> reduce_counter(reduce_shape.size(), 0);
            // std::array<char*, ScalarTypesSize> data_ptrs;
            // std::array<index_t, ScalarTypesSize> data_strides;
            //
            // #pragma omp parallel for default(none) \
            //     shared(non_reduce_size, reduce_size, reduce_func, operands_, shape_, is_reduce_dim) \
            //     private(non_reduce_counter, reduce_counter, data_ptrs, data_strides) \
            //     schedule(dynamic, 64)
            // for (index_t i = 0; i < non_reduce_size; ++i) {
            //     // 设置非 reduce 维度的索引
            //     index_to_counter(i, non_reduce_counter, non_reduce_shape);
            //
            //     // 初始化 reduce 操作的结果
            //     // 注意：这里假设第一个操作数是输出
            //     char* out_ptr = operands_[0].data + operands_[0].begin_offset * ResultTypeSize +
            //         compute_offset(non_reduce_counter, operands_[0].strides_bytes);
            //
            //     // 对 reduce 维度进行循环
            //     for (index_t j = 0; j < reduce_size; ++j) {
            //         index_to_counter(j, reduce_counter, reduce_shape);
            //
            //         // 设置数据指针和步长
            //         for (size_t op = 0; op < ScalarTypesSize; ++op) {
            //             std::vector<index_t> full_counter;
            //             Utils::merge_counters(non_reduce_counter, reduce_counter, is_reduce_dim, full_counter);
            //
            //             data_ptrs[op] = operands_[op].data + operands_[op].begin_offset * ResultTypeSize +
            //                 compute_offset(full_counter, operands_[op].strides_bytes);
            //
            //             // 对于 reduce 操作，我们可能需要调整步长的计算方式
            //             data_strides[op] = compute_reduce_stride(operands_[op].strides, is_reduce_dim);
            //         }
            //
            //         // 调用 reduce_impl
            //         reduce_impl(out_ptr, data_ptrs.data() + 1, data_strides.data() + 1, 1, reduce_func);
            //     }
            // }
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

        template<typename VectorOp, size_t... Indices>
        static void vectorized_loop_impl(VectorOp& op, const std::array<char*, ScalarTypesSize>& data_ptrs,
                              char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<Indices...>) {

            constexpr index_t cache_line_size = 64; // 缓存行大小为64字节
            constexpr index_t prefetch_elements = cache_line_size / sizeof(ResultScalarType); // 以元素数量计的预读取数量

            for (index_t i = 0; i <= size - simd_vector_size; i += simd_vector_size) {
                // 执行向量化操作
                op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                   Vectorized<ResultScalarType>::loadu(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i)...);
                // 每处理 prefetch_elements 个向量后进行一次预读取
                if (i % prefetch_elements == 0 && i < size - prefetch_elements) {
                    Vectorized<ResultScalarType>::prefetch(reinterpret_cast<ResultScalarType*>(output_ptr) + i + prefetch_elements);
                    (Vectorized<ResultScalarType>::prefetch(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i + prefetch_elements), ...);
                }
            }
        }


        template<typename ScalarOp, size_t... Indices>
            static void scalar_loop_impl(ScalarOp& op, const std::array<char*, ScalarTypesSize> &data_ptrs,
                char* output_ptr, const index_t size, const index_t simd_vector_size, std::index_sequence<Indices...>) {
            const auto begin = size - size % simd_vector_size;
            for (index_t i = begin; i < size; i += 1) {
                //这里 不同类型转换使用默认编译器规则
                op(reinterpret_cast<ResultScalarType*>(output_ptr) + i,
                   *(reinterpret_cast<ResultScalarType*>(data_ptrs[Indices]) + i)...);
            }
        }

        template <typename ReduceOp>
        void reduce_impl(TensorIterator& iter, const ReduceOp& reduce_op) {
            auto loop = [&](char** data, const int64_t* strides, const int64_t n) {
                auto out = static_cast<ResultScalarType*>(data[0]);
                auto in =  static_cast<ResultScalarType*>(data[1]);

                const int64_t out_stride = strides[0];
                const int64_t in_stride = strides[1];

                for (int64_t i = 0; i < n; i++) {
                    *out = reduce_op(*out, *in);
                    //计算下一个reduce 的地址
                    out += out_stride;
                    in += in_stride;
                }
            };
            iter.for_each(loop);
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

        template<size_t... Indices>
        void check_safe_to_output(std::index_sequence<Indices...>) const {
            std::array<index_t, sizeof...(Indices)> data_size = { get_tensor_data_size<Indices>()...};
            const index_t output_size = data_size[sizeof...(Indices) - 1];
            if (const bool is_safe = ((Indices == sizeof...(Indices) - 1 || output_size >= data_size[Indices]) && ...); !is_safe) {
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
            if (strides.size() <= 1) return true;
            index_t expected_stride = 1;
            for (index_t i = static_cast<index_t>(shape.size()) - 1; i >= 0; --i) {
                if (shape[i] == 1) continue;
                if (strides[i] != expected_stride) return false;
                expected_stride *= shape[i];
            }
            return true;
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
            if constexpr (sizeof...(Indices) == 0) {
                return {};
            } else if constexpr (sizeof...(Indices) == 1) {
                return get_shape_from_tuple<0>();
            } else {
                return Utils::broadcast_shapes(get_shape_from_tuple<Indices>()...);
            }
        }

        void permute_dimensions(std::vector<index_t>& perm) {
            assert(perm.size() == static_cast<unsigned>(ndim()));
            auto reorder = [&](const auto& vec) {
                auto res = vec;
                for (size_t i = 0; i < perm.size(); i++) {
                    res[i] = vec[perm[i]];
                }
                return res;
            };
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