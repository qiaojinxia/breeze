#include "CPUTensorOps.h"
#include <cblas.h>
#include "TensorIterator.h"
#include "platform/SIMDFactory.h"
#include "CPUTensor.h"

namespace Breeze {

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
        return nullptr;
    }

    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
    template class CPUTensorOps<float, float>;
    template class CPUTensorOps<float, double>;
    template class CPUTensorOps<double, double>;
    template class CPUTensorOps<double, float>;

} // namespace Breeze