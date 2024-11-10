#include "CPUTensorOps.h"
#include <cblas.h>
#include "TensorIterator.h"
#include "platform/SIMD.h"
#include "CPUTensor.h"
#include "./lib/pcg/pcg_random.hpp"

namespace Breeze {
    template<typename ScalarT, index_t VecSize>
    struct ArangeIndices {
        static constexpr std::array<ScalarT, VecSize> values = []{
            std::array<ScalarT, VecSize> arr{};
            for (int i = 0; i < VecSize; ++i) {
                arr[i] = static_cast<ScalarT>(i);
            }
            return arr;
        }();
    };

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::fill(Tensor<ScalarT1> &a, ScalarT1 value) const {
        using ScalarT1 = typename BaseOps::ScalarT1;
        auto iter = TensorIterator<ScalarT1>::nullary_op(a);
        iter.cpu_kernel_vec(
            [value](ScalarT1 *out_ptr) {
                *out_ptr = static_cast<ScalarT1>(value);
            },
            [value](ScalarT1 *out_ptr) {
                auto out_vec = Vectorized<ScalarT1>(value);
                out_vec.store(out_ptr);
            }

        );
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::arange(Tensor<ScalarT1> &a, const ScalarT1 start,const ScalarT1 step) const {
        using ScalarT1 = typename BaseOps::ScalarT1;
        auto iter = TensorIterator<ScalarT1>::nullary_op(a);
        iter.cpu_kernel_vec(
            [&a, start, step](ScalarT1 *out_ptr) {
                auto offset_index = out_ptr - a.mutable_data();
                *out_ptr = offset_index * step + start;
            },
            [&a, start, step](ScalarT1 *out_ptr) {
                auto offset_index = out_ptr - a.mutable_data();
                constexpr int vec_size = Vectorized<ScalarT1>::size();
                constexpr auto inc_indices = ArangeIndices<ScalarT1, vec_size>::values;
                auto inc_indices_vec = Vectorized<ScalarT1>::loadu(inc_indices.data());
                auto start_vec = Vectorized<ScalarT1>(start);
                auto offset_vec = Vectorized<ScalarT1>(offset_index);
                auto step_vec = Vectorized<ScalarT1>(step);
                Vectorized<ScalarT1> out_vec = start_vec + (inc_indices_vec + offset_vec ) * step_vec;
                out_vec.store(out_ptr);
            }
        );
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::randn(Tensor<ScalarT1> &a) const {
        using ScalarT1 = typename BaseOps::ScalarT1;
        using RealType = std::conditional_t<std::is_same_v<ScalarT1, float> || std::is_same_v<ScalarT1, double>, ScalarT1, float>;

        auto iter = TensorIterator<ScalarT1>::nullary_op(a);
        alignas(64) pcg_extras::seed_seq_from<std::random_device> seed_source;

        pcg32 rng(seed_source);
        std::normal_distribution<RealType> dist(0.0, 1.0);
        std::uniform_real_distribution<RealType> uniform(0.0, 1.0);

        iter.cpu_kernel_vec(
            [&dist, &rng](ScalarT1 *out_ptr) {
                *out_ptr = static_cast<ScalarT1>(dist(rng));
            },
            [&uniform, &rng](ScalarT1 *out_ptr) {
                constexpr int vec_size = Vectorized<RealType>::size();
                alignas(32) std::array<RealType, vec_size> u1{};
                alignas(32) std::array<RealType, vec_size> u2{};

                for (int i = 0; i < vec_size; ++i) {
                    u1[i] = uniform(rng);
                    u2[i] = uniform(rng);
                }

                auto vec_1 = Vectorized<RealType>::loadu(reinterpret_cast<RealType *>(&u1[0]));
                auto vec_2 = Vectorized<RealType>::loadu(reinterpret_cast<RealType *>(&u2[0]));
                auto out_vec = Vectorized<RealType>::randn(vec_1, vec_2);

                // 如果ScalarT1不是float/double,需要转换
                if constexpr (!std::is_same_v<ScalarT1, RealType>) {
                    alignas(32) std::array<RealType, vec_size> temp{};
                    out_vec.store(temp.data());
                    for(int i = 0; i < vec_size; ++i) {
                        out_ptr[i] = static_cast<ScalarT1>(temp[i]);
                    }
                } else {
                    out_vec.store(out_ptr);
                }
            }
        );
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::sum(
        const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            []{return ResultT(0);},
            [](ResultT *out_ptr, ResultT a_value) {
                 *out_ptr += a_value;
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                 auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                 auto sum_vec = a_vec + out_vec;
                 sum_vec.store(out_ptr);
            },
            [](const ResultT* data, const index_t size) {
                 ResultT sum_val = data[0];
                 for (index_t i = 1; i < size; ++i) {
                      sum_val += data[i];
                 }
                 return sum_val;
             }
         );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::prod(
        const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            []{return ResultT(1.0);},
            [](ResultT *out_ptr, ResultT a_value) {
                 *out_ptr *= a_value;
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                 auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                 auto sum_vec = a_vec * out_vec;
                 sum_vec.store(out_ptr);
            },
            [](const ResultT* data, const index_t size) {
                 ResultT sum_val = data[0];
                 for (index_t i = 1; i < size; ++i) {
                      sum_val *= data[i];
                 }
                 return sum_val;
             }
         );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::max(
      const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            []{return ResultT(std::numeric_limits<ResultT>::min());},
            [](ResultT *out_ptr, ResultT a_value) {
                  *out_ptr = std::max(*out_ptr, a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                 auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                 auto max_vec = Vectorized<ResultT>::max(out_vec, a_vec);
                 max_vec.store(out_ptr);
            },
            [](const ResultT* data, const index_t size) {
                 ResultT max_val = data[0];
                 for (index_t i = 1; i < size; ++i) {
                      max_val = std::max(max_val, data[i]);
                 }
                 return max_val;
            }
         );
        return result;
    }

    template <typename ScalarType, typename IndexType>
    struct MaxArgData {
        IndexType begin_index;
        Vectorized<ScalarType> max_vec;
        Vectorized<IndexType> max_index_vec;

        MaxArgData() : begin_index(0),
            max_vec(std::numeric_limits<IndexType>::lowest()),
            max_index_vec(0) {}  // 直接使用0即可,不需要ScalarT1
    };

    template <typename ... ScalarTypes>
    std::shared_ptr<Tensor<i64>> CPUTensorOps<ScalarTypes...>::arg_max(const Tensor<ScalarT1>& a,
        std::vector<index_t>& dims, const bool keep_dim) const
    {
        auto result = std::make_shared<CPUTensor<index_t>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
        .set_resize_outputs(true)
        .set_reduce_dims(dims)
        .set_is_reduction(true)
        .set_keep_keep_dim(keep_dim);

        using IndexType = std::conditional_t<std::is_same_v<ScalarT1, float>, i32, i64>;
        auto iter = TensorIterator<ScalarT1, i64>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
           []{
               return MaxArgData<ScalarT1, IndexType>();
           },
           [](MaxArgData<ScalarT1, IndexType> *data, const ScalarT1 a_value) {
               if (a_value > data->max_vec[0])
               {
                   data->max_vec[0] = a_value;
                   data->max_index_vec[0] = data->begin_index;
                   ++data->begin_index;
               }
           },
           [](MaxArgData<ScalarT1,IndexType>* data, const Vectorized<ScalarT1>& a_vec) {
               //生成当前索引向量 (0,1,2,3,....) + begin_index
               Vectorized<IndexType> curr_indices = Vectorized<IndexType>::arange(data->begin_index);
               // 比较并生成mask
               auto mask = a_vec > data->max_vec;
               // 更新最大值
               data->max_vec = Vectorized<ScalarT1>::blendv(a_vec, data->max_vec, mask);
               // 更新对应的索引
               data->max_index_vec = Vectorized<IndexType>::blendv(data->max_index_vec,
                   curr_indices, Vectorized<IndexType>(mask.get_values()));
               // 更新begin_index为下一组
               data->begin_index += Vectorized<ScalarT1>::size();
           },
           [](const MaxArgData<ScalarT1,IndexType>* data, const index_t size){
               IndexType max_index = data->max_index_vec[0];
               ScalarT1 max_vec = data->max_vec[0];
               for (index_t i = 1; i < size; ++i){
                   if(max_vec < data->max_vec[i]){
                       max_vec = data->max_vec[i];
                       max_index = data->max_index_vec[i];
                   }
               }
               return max_index;
           });
        return result;
    }


    template <typename ScalarType, typename IndexType>
    struct MinArgData {
        IndexType begin_index;
        Vectorized<ScalarType> min_vec;
        Vectorized<IndexType> min_index_vec;

        MinArgData() : begin_index(0),
            min_vec(std::numeric_limits<ScalarType>::max()),
            min_index_vec(0) {}
    };

    template <typename ... ScalarTypes>
    std::shared_ptr<Tensor<i64>> CPUTensorOps<ScalarTypes...>::arg_min(const Tensor<ScalarT1>& a,
        std::vector<index_t>& dims, const bool keep_dim) const
    {
        auto result = std::make_shared<CPUTensor<index_t>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);

        using IndexType = std::conditional_t<std::is_same_v<ScalarT1, float>, i32, i64>;
        auto iter = TensorIterator<ScalarT1, i64>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            []{
                return MinArgData<ScalarT1, IndexType>();
            },
            [](MinArgData<ScalarT1, IndexType> *data, const ScalarT1 a_value) {
                if (a_value < data->min_vec[0])
                {
                    data->min_vec[0] = a_value;
                    data->min_index_vec[0] = data->begin_index;
                    ++data->begin_index;
                }
            },
            [](MinArgData<ScalarT1, IndexType>* data, const Vectorized<ScalarT1>& a_vec) {
                Vectorized<IndexType> curr_indices = Vectorized<IndexType>::arange(data->begin_index);
                auto mask = a_vec < data->min_vec;
                // 更新最小值
                data->min_vec = Vectorized<ScalarT1>::blendv(data->min_vec, a_vec, mask);
                // 更新对应的索引
                data->min_index_vec = Vectorized<IndexType>::blendv(data->min_index_vec,
                    curr_indices, Vectorized<IndexType>(mask.get_values()));
                data->begin_index += Vectorized<ScalarT1>::size();
            },
            [](const MinArgData<ScalarT1,IndexType>* data, const index_t size){
                IndexType min_index = data->min_index_vec[0];
                ScalarT1 min_vec = data->min_vec[0];
                for (index_t i = 1; i < size; ++i){
                    if(min_vec > data->min_vec[i]){
                        min_vec = data->min_vec[i];
                        min_index = data->min_index_vec[i];
                    }
                }
                return min_index;
            });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::min(
      const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        // 获取 ResultT 类型的最大可能值
        iter.reduce_strided_for_each(
            []{return ResultT(std::numeric_limits<ResultT>::max());},
            [](ResultT *out_ptr, ResultT a_value) {
                  *out_ptr = std::min(*out_ptr, a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                 auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                 auto min_vec = Vectorized<ResultT>::min(out_vec, a_vec);
                 min_vec.store(out_ptr);
            },
            [](const ResultT* data, const index_t size) {
                 ResultT min_val = data[0];
                 for (index_t i = 1; i < size; ++i) {
                      min_val = std::min(min_val, data[i]);
                 }
                 return min_val;
            }
         );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::mean(
        const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        // 计算总元素数
        index_t total_elements = 1;
        for (const auto dim : dims ) {
            total_elements *= a.get_shape().dims()[dim];
        }
        const TensorIteratorConfig config = TensorIteratorConfig()
               .set_resize_outputs(true)
               .set_reduce_dims(dims)
               .set_is_reduction(true)
               .set_keep_keep_dim(keep_dim);

        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            []{return ResultT(0);},
            [](ResultT *out_ptr, ResultT a_value) {
                *out_ptr += a_value;
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                auto sum_vec = out_vec + a_vec;
                sum_vec.store(out_ptr);
            },
            [total_elements](const ResultT* data, const index_t size) {
                ResultT sum_val = data[0];
                for (index_t i = 1; i < size; ++i) {
                    sum_val += data[i];
                }
                return sum_val / static_cast<ResultT>(total_elements);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::std(
        const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim, const bool unbiased) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto cp_dims = std::vector<index_t>(dims.begin(), dims.end());

        // 计算总元素数
        index_t total_elements = 1;
        for (const auto dim : dims) {
            total_elements *= a.get_shape().dims()[dim];
        }

        // 调整分母
        ResultT correction = unbiased ? total_elements - 1 : total_elements;
        correction = static_cast<ResultT>(1) / correction;

        // 配置迭代器
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);

        // 创建迭代器
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        // 小数据量使用优化后的两遍算法
        struct TwoPassData {
            ResultT mean = 0;
            ResultT sum_sq = 0;
            index_t count = 0;
        };
        // 一次遍历计算均值和平方和
        iter.reduce_strided_for_each(
            []{return TwoPassData();},
            [](TwoPassData* data, ResultT val) {
                data->mean += val;
                data->sum_sq += val * val;
                ++data->count;
            },
            [](TwoPassData* data, const Vectorized<ResultT> vec) {
                auto pow_vec = vec * vec;
                data->mean += vec.horizontal_sum();
                data->sum_sq += pow_vec.horizontal_sum();
                data->count += Vectorized<ResultT>::size();
            },
            [correction](const TwoPassData* data, const index_t size) {
                (void)size;
                ResultT mean = data->mean / data->count;
                // 使用公式: Var(X) = E(X^2) - E(X)^2
                ResultT variance = (data->sum_sq / data->count) - (mean * mean);
                variance *= correction * data->count;  // 应用修正因子
                return std::sqrt(variance);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::var(
        const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const bool keep_dim, const bool unbiased) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto cp_dims = std::vector<index_t>(dims.begin(), dims.end());

        // 计算总元素数
        index_t total_elements = 1;
        for (const auto dim : dims) {
            total_elements *= a.get_shape().dims()[dim];
        }

        // 调整分母（无偏估计时使用 n-1）
        ResultT correction = unbiased ? total_elements - 1 : total_elements;
        correction = static_cast<ResultT>(1) / correction;

        // 配置迭代器
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_reduce_dims(dims)
            .set_is_reduction(true)
            .set_keep_keep_dim(keep_dim);

        // 创建迭代器
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);

        // 定义两遍算法的数据结构
        struct TwoPassData {
            ResultT mean = 0;
            ResultT sum_sq = 0;
            index_t count = 0;
        };

        // 一次遍历计算均值和平方和
        iter.reduce_strided_for_each(
            // 初始化函数
            []{return TwoPassData();},
            // 标量处理函数
            [](TwoPassData* data, ResultT val) {
                data->mean += val;
                data->sum_sq += val * val;
                ++data->count;
            },

            // 向量化处理函数
            [](TwoPassData* data, const Vectorized<ResultT> vec) {
                auto pow_vec = vec * vec;
                data->mean += vec.horizontal_sum();
                data->sum_sq += pow_vec.horizontal_sum();
                data->count += Vectorized<ResultT>::size();
            },

            // 最终计算函数
            [correction](const TwoPassData* data, const index_t size) {
                (void)size;
                ResultT mean = data->mean / data->count;
                // 使用公式: Var(X) = E(X^2) - E(X)^2
                ResultT variance = (data->sum_sq / data->count) - (mean * mean);
                return variance * correction * data->count;
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::norm(
    const Tensor<ScalarT1> &a, std::vector<index_t> &dims, const int p, const bool keep_dim) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        // 配置迭代器
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(true)
            .set_is_reduction(true)
            .set_reduce_dims(dims)
            .set_keep_keep_dim(keep_dim);
        // 创建迭代器
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        // 特殊处理无穷范数
        if (p == INF) {
            // 使用迭代器计算无穷范数
            iter.reduce_strided_for_each(
                []{return ResultT(std::numeric_limits<ResultT>::lowest());},
                // 找到每个元素的绝对值最大值
                [](ResultT *out_ptr, ResultT a_value) {
                    *out_ptr = std::max(*out_ptr, std::abs(a_value));
                },
                [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                    auto abs_max_vec = a_vec.abs();
                    auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                    auto max_vec = Vectorized<ResultT>::max(out_vec, abs_max_vec);
                    max_vec.store(out_ptr);
                },
                // 无需进一步处理，因为已经找到了最大值
                [](const ResultT* data, const index_t size) {
                    ResultT max_val = data[0];
                    for (index_t i = 1; i < size; ++i) {
                        max_val = std::max(max_val, data[i]);
                    }
                    return max_val;
                }
            );
        } else {
            // 使用迭代器计算范数
            iter.reduce_strided_for_each(
                []{return ResultT(0);},
                // 累加每个元素的p次幂
                [p](ResultT *out_ptr, ResultT a_value) {
                    *out_ptr += std::pow(std::abs(a_value), p);
                },
                [p](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                    auto p_vec = Vectorized<ResultT>(static_cast<ResultT>(p));
                    auto abs_pow_vec = a_vec.abs().pow(p_vec);
                    auto out_vec = Vectorized<ResultT>::loadu(out_ptr);
                    auto sum_vec = out_vec + abs_pow_vec;
                    sum_vec.store(out_ptr);
                },
                // 取p次幂的p次根得到范数
                [p](const ResultT* data, const index_t size) {
                    ResultT sum_pow = data[0];
                    for (index_t i = 1; i < size; ++i) {
                        sum_pow += data[i];
                    }
                    return std::pow(sum_pow, 1.0 / p);
                }
            );
        }
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::sin(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr, ResultT a_value) {
                *out_ptr = std::sin(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.sin();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::cos(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::cos(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.cos();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::tan(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::tan(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.tan();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::atan(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::atan(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.atan();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::log(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::log(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.log();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::log2(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::log2(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.log2();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::log10(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::log10(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.log10();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::exp(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::exp(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.exp();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::sqrt(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::sqrt(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.sqrt();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::rsqrt(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = 1 / std::sqrt(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                auto ones_vec = Vectorized<ResultT>(1.0);
                Vectorized<ResultT> out_vec = ones_vec / a_vec.sqrt();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }


    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarT1>> CPUTensorOps<ScalarTypes...>::abs(
        const Tensor<ScalarT1> &a) const {
        using ResultT = ScalarT1;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
                *out_ptr = std::abs(a_value);
            },
            [](ResultT* out_ptr, const Vectorized<ResultT> a_vec) {
                Vectorized<ResultT> out_vec = a_vec.abs();
                out_vec.store(out_ptr);
            }
        );
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::pow(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
           [](ResultT* out_ptr, const ResultT a_value, const ResultT b_value) {
               *out_ptr = std::pow(a_value, b_value);
           },
           [](ResultT* out_ptr, const Vectorized<ResultT> a_vec, const Vectorized<ResultT> b_vec) {
               Vectorized<ResultT> out_vec = a_vec.pow(b_vec);
               out_vec.store(out_ptr);
       });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::add(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
           [](ResultT* out_ptr, const ResultT a_value, const ResultT b_value) {
               *out_ptr = a_value + b_value;
           },
           [](ResultT* out_ptr, const Vectorized<ResultT> a_vec, const Vectorized<ResultT> b_vec) {
               Vectorized<ResultT> out_vec = a_vec + b_vec;
               out_vec.store(out_ptr);
       });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::subtract(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultType = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultType>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
        //判断是否标量 得到标量的值 然后 直接把标量值操作
        iter.cpu_kernel_vec(
           [](ResultType* out_ptr, const ResultType a_value, const ResultType b_value) {
               *out_ptr = a_value - b_value;
           },
           [](ResultType* out_ptr, const Vectorized<ResultType> a_vec, const Vectorized<ResultType> b_vec) {
               Vectorized<ResultType> out_vec = a_vec - b_vec;
               out_vec.store(out_ptr);
       });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::divide(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
           [](ResultT* out_ptr, const ResultT a_value, const ResultT b_value) {
               *out_ptr = a_value / b_value;
           },
           [](ResultT* out_ptr, const Vectorized<ResultT> a_vec, const Vectorized<ResultT> b_vec) {
               Vectorized<ResultT> out_vec = a_vec / b_vec;
               out_vec.store(out_ptr);
       });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::multiply(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
           [](ResultT* out_ptr, const ResultT a_value, const ResultT b_value) {
               *out_ptr = a_value * b_value;
           },
           [](ResultT* out_ptr, const Vectorized<ResultT> a_vec, const Vectorized<ResultT> b_vec) {
               Vectorized<ResultT> out_vec = a_vec * b_vec;
               out_vec.store(out_ptr);
       });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::ScalarResult>> CPUTensorOps<ScalarTypes...>::matmul(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
         // Get shapes of tensors a and b
        // const std::vector<index_t> a_shape = a.get_shape().dims();
        // const std::vector<index_t> b_shape = b.get_shape().dims();
        // using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        //
        // // Check for correct dimensions
        // if (a_shape.size() < 2 || b_shape.size() < 2) {
        //     throw std::invalid_argument("Input tensors must have at least two dimensions for matrix multiplication.");
        // }
        //
        // // Check if inner dimensions match
        // if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
        //     throw std::invalid_argument("The inner dimensions must match for matrix multiplication.");
        // }
        //
        // // Calculate the broadcast shape
        // auto [a_strides, b_strides, result_shape] =
        //     Utils::calc_matmul_shape(a_shape, b_shape);
        //
        // // Allocate result tensor
        // auto result = std::make_shared<CPUTensor<ResultT>>(Shape{result_shape});
        //
        // // Compute the strides for each tensor
        // const std::vector<index_t> result_strides = result->get_strides();
        //
        // const index_t depth = static_cast<index_t>(result_shape.size()) - 2;
        // const index_t m = a_shape[a_shape.size() - 2];
        // const index_t k = a_shape[a_shape.size() - 1];
        // const index_t n = b_shape[b_shape.size() - 1];
        //
        // CBLAS_ORDER order = CblasRowMajor;
        // CBLAS_TRANSPOSE transA = CblasNoTrans;
        // CBLAS_TRANSPOSE transB = CblasNoTrans;
        // auto alpha = static_cast<ResultT>(1.0);
        // auto beta = static_cast<ResultT>(0.0);
        //
        // // Calculate the number of 2D matrices
        // index_t num_matrices = 1;
        // for (index_t i = 0; i < depth; ++i) {
        //     num_matrices *= result_shape[i];
        // }
        //
        // const ScalarT1* a_data = a.data();
        // const ScalarT2* b_data = b.data();
        // ResultT* result_data = result->mutable_data();
        //
        // for (index_t idx = 0; idx < num_matrices; ++idx) {
        //     std::vector<index_t> coords(depth);
        //     index_t temp = idx;
        //     for (index_t i = static_cast<int>(depth) - 1; i >= 0; --i) {
        //         coords[i] = temp % result_shape[i];
        //         temp /= result_shape[i];
        //     }
        //
        //     index_t a_offset = a.get_offset(), b_offset = b.get_offset(), result_offset = 0;
        //     for (index_t i = 0; i < depth; ++i) {
        //         a_offset += (coords[i] % a_shape[i]) * a_strides[i];
        //         b_offset += (coords[i] % b_shape[i]) * b_strides[i];
        //         result_offset += coords[i] * result_strides[i];
        //     }
        //
        //     const ScalarT1* a_sub = a_data + a_offset;
        //     const ScalarT2* b_sub = b_data + b_offset;
        //     ResultT* result_sub = result_data + result_offset;
        //
        //     // Use BLAS for matrix multiplication
        //     if constexpr (std::is_same_v<ResultT, float>) {
        //         // cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
        //     } else if constexpr (std::is_same_v<ResultT, double>) {
        //         // cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
        //     } else {
        //         // Fallback scalar implementation
        //         for (index_t i = 0; i < m; ++i) {
        //             for (index_t j = 0; j < n; ++j) {
        //                 ResultT sum = 0;
        //                 for (index_t l = 0; l < k; ++l) {
        //                     sum += a_sub[i * k + l] * b_sub[l * n + j];
        //                 }
        //                 result_sub[i * n + j] = sum;
        //             }
        //         }
        //     }
        // }
        return nullptr;
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::add_inplace(Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        const TensorIteratorConfig config = TensorIteratorConfig()
            .set_resize_outputs(false)
            .set_enforce_safe_casting_to_output(true);
        auto iter = TensorIterator<ScalarT1, ScalarT2>::unary_op(a, b, config);
        iter.cpu_kernel_vec(
            [](ScalarT1 *out_ptr, ScalarT1 a_value) {
                *out_ptr = *out_ptr + a_value;
            },
            [](ScalarT1* out_ptr, const Vectorized<ScalarT1> b_vec) {
                auto a_vec = Vectorized<ScalarT1>::loadu(out_ptr);
                Vectorized<ScalarT1> out_vec = a_vec + b_vec;
                out_vec.store(out_ptr);
            }
        );
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::subtract_inplace(Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        auto iter = TensorIterator<ScalarT1, ScalarT1>::unary_op(a, b);
        iter.cpu_kernel_vec(
            [](ScalarT1 *out_ptr, ScalarT1 a_value) {
                *out_ptr = *out_ptr - a_value;
            },
            [](ScalarT1* out_ptr, const Vectorized<ScalarT1> b_vec) {
                auto a_vec = Vectorized<ScalarT1>::loadu(out_ptr);
                Vectorized<ScalarT1> out_vec = a_vec - b_vec;
                out_vec.store(out_ptr);
            }
        );
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::multiply_inplace(Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        auto iter = TensorIterator<ScalarT1, ScalarT2>::unary_op(a, b);
        iter.cpu_kernel_vec(
            [](ScalarT1 *out_ptr,ScalarT1 a_value) {
                *out_ptr = *out_ptr * a_value;
            },
            [](ScalarT1* out_ptr, const Vectorized<ScalarT1> b_vec) {
                auto a_vec = Vectorized<ScalarT1>::loadu(out_ptr);
                Vectorized<ScalarT1> out_vec = a_vec * b_vec;
                out_vec.store(out_ptr);
            }
        );
    }

    template<typename ... ScalarTypes>
    void CPUTensorOps<ScalarTypes...>::divide_inplace(Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        auto iter = TensorIterator<ScalarT1, ScalarT2>::unary_op(a, b);
        iter.cpu_kernel_vec(
            [](ScalarT1 *out_ptr, ScalarT1 a_value) {
                *out_ptr = *out_ptr / a_value;
            },
            [](ScalarT1* out_ptr, const Vectorized<ScalarT1> b_vec) {
                auto a_vec = Vectorized<ScalarT1>::loadu(out_ptr);
                Vectorized<ScalarT1> out_vec = a_vec / b_vec;
                out_vec.store(out_ptr);
            }
        );
    }

    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
    template class CPUTensorOps<i64>;
    template class CPUTensorOps<float, i64>;
    template class CPUTensorOps<float, float>;
    template class CPUTensorOps<float, double>;
    template class CPUTensorOps<double, i64>;
    template class CPUTensorOps<double, float>;
    template class CPUTensorOps<double, double>;
    template class CPUTensorOps<i64, i64>;
    template class CPUTensorOps<i64, float>;
    template class CPUTensorOps<i64, double>;
} // namespace Breeze