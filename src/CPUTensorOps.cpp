#include "CPUTensorOps.h"
#include <cblas.h>
#include "TensorIterator.h"
#include "platform/SIMDFactory.h"
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
        auto iter = TensorIterator<ScalarT1>::nullary_op(a);
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        // 创建 PCG 随机数生成器
        pcg32 rng(seed_source);
        std::normal_distribution<ScalarT1> dist(0.0, 1.0);
        // 创建 Box-Muller 变换所需的均匀分布
        std::uniform_real_distribution<ScalarT1> uniform(0.0, 1.0);
        iter.cpu_kernel_vec(
            [&dist, &rng](ScalarT1 *out_ptr) {
                *out_ptr = dist(rng);
            },
            [&uniform, &rng](ScalarT1 *out_ptr) {
                // SIMD 版本：使用 Box-Muller 变换
               constexpr int vec_size = Vectorized<ScalarT1>::size();
               alignas(32) std::array<ScalarT1, vec_size> u1{};
               alignas(32) std::array<ScalarT1, vec_size> u2{};

               // 生成均匀分布的随机数
               for (int i = 0; i < vec_size; ++i) {
                   u1[i] = uniform(rng);
                   u2[i] = uniform(rng);
               }
                auto vec_1 = Vectorized<ScalarT1>::loadu(reinterpret_cast<ScalarT1 *>(&u1[0]));
                auto vec_2 = Vectorized<ScalarT1>::loadu(reinterpret_cast<ScalarT1 *>(&u2[0]));
                auto out_vec = Vectorized<ScalarT1>::randn(vec_1, vec_2);
                out_vec.store(out_ptr);
            }
        );
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::sum(
        Tensor<ScalarT1> &a, std::vector<index_t> &dims) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();

        const TensorIteratorConfig config = TensorIteratorConfig()
        .set_resize_outputs(true)
        .set_reduce_dims(dims)
        .set_is_reduction(true)
        .set_keep_keepdim(true);
        auto iter = TensorIterator<ResultT, ResultT>::reduce_op(*result, a, config);
        iter.reduce_strided_for_each(
            [](ResultT a_value, ResultT b_value) {
                return a_value + b_value;
        });
        return result;
    }

    template<typename ... ScalarTypes>
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::sin(
        const Tensor<ScalarT1> &a) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultT>>();
        auto iter = TensorIterator<ResultT, ResultT>::unary_op(*result, a);
        iter.cpu_kernel_vec(
            [](ResultT *out_ptr,ResultT a_value) {
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::cos(
        const Tensor<ScalarT1> &a) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::tan(
        const Tensor<ScalarT1> &a) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::atan(
        const Tensor<ScalarT1> &a) const {
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::pow(
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::add(
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::subtract(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
        using ResultType = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;
        auto result = std::make_shared<CPUTensor<ResultType>>();
        auto iter = TensorIterator<ScalarT1, ScalarT2>::binary_op(*result, a, b);
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::divide(
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::multiply(
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
    std::shared_ptr<Tensor<typename CPUTensorOps<ScalarTypes...>::scalar_result>> CPUTensorOps<ScalarTypes...>::matmul(
        const Tensor<ScalarT1> &a, const Tensor<ScalarT2> &b) const {
         // Get shapes of tensors a and b
        const std::vector<index_t> a_shape = a.get_shape().dims();
        const std::vector<index_t> b_shape = b.get_shape().dims();
        using ResultT = typename BinaryOpResultType<ScalarT1, ScalarT2>::type;

        // Check for correct dimensions
        if (a_shape.size() < 2 || b_shape.size() < 2) {
            throw std::invalid_argument("Input tensors must have at least two dimensions for matrix multiplication.");
        }

        // Check if inner dimensions match
        if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
            throw std::invalid_argument("The inner dimensions must match for matrix multiplication.");
        }

        // Calculate the broadcast shape
        auto [a_strides, b_strides, result_shape] =
            Utils::calc_matmul_shape(a_shape, b_shape);

        // Allocate result tensor
        auto result = std::make_shared<CPUTensor<ResultT>>(Shape{result_shape});

        // Compute the strides for each tensor
        const std::vector<index_t> result_strides = result->get_strides();

        const index_t depth = static_cast<index_t>(result_shape.size()) - 2;
        const index_t m = a_shape[a_shape.size() - 2];
        const index_t k = a_shape[a_shape.size() - 1];
        const index_t n = b_shape[b_shape.size() - 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        auto alpha = static_cast<ResultT>(1.0);
        auto beta = static_cast<ResultT>(0.0);

        // Calculate the number of 2D matrices
        index_t num_matrices = 1;
        for (index_t i = 0; i < depth; ++i) {
            num_matrices *= result_shape[i];
        }

        const ScalarT1* a_data = a.data();
        const ScalarT2* b_data = b.data();
        ResultT* result_data = result->mutable_data();

        for (index_t idx = 0; idx < num_matrices; ++idx) {
            std::vector<index_t> coords(depth);
            index_t temp = idx;
            for (index_t i = static_cast<int>(depth) - 1; i >= 0; --i) {
                coords[i] = temp % result_shape[i];
                temp /= result_shape[i];
            }

            index_t a_offset = a.get_offset(), b_offset = b.get_offset(), result_offset = 0;
            for (index_t i = 0; i < depth; ++i) {
                a_offset += (coords[i] % a_shape[i]) * a_strides[i];
                b_offset += (coords[i] % b_shape[i]) * b_strides[i];
                result_offset += coords[i] * result_strides[i];
            }

            const ScalarT1* a_sub = a_data + a_offset;
            const ScalarT2* b_sub = b_data + b_offset;
            ResultT* result_sub = result_data + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<ResultT, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<ResultT, double>) {
                // cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else {
                // Fallback scalar implementation
                for (index_t i = 0; i < m; ++i) {
                    for (index_t j = 0; j < n; ++j) {
                        ResultT sum = 0;
                        for (index_t l = 0; l < k; ++l) {
                            sum += a_sub[i * k + l] * b_sub[l * n + j];
                        }
                        result_sub[i * n + j] = sum;
                    }
                }
            }
        }
        return result;
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
        auto iter = TensorIterator<ScalarT1, ScalarT2>::unary_op(a, b);
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
    template class CPUTensorOps<float, float>;
    template class CPUTensorOps<float, double>;
    template class CPUTensorOps<double, double>;
    template class CPUTensorOps<double, float>;

} // namespace Breeze