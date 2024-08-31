// Created by mac on 2024/7/31.
//
#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>
#include "TensorIterator.h"
#include "platform/SIMDFactory.h"

namespace Breeze {
    template<typename T>
    void CPUTensorOps<T>::fill(Tensor<T>& a, T value) const {
        TensorIterator<T> iter;
        iter.add_output(a);
        iter.build();
        iter.cpu_kernel_vec(
        [&](T* out_ptr) {
            *out_ptr = value;
        },
        [&](T* out_ptr) {
            auto value_vec = Vectorized<T>(value);
            value_vec.store(out_ptr);
        });
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::add(const Tensor<T>& a, const Tensor<T>& b) const {
        auto result = std::make_shared<CPUTensor<T>>();
        auto iter = TensorIterator<T>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
        [](T* out_ptr, const T* a_ptr, const T* b_ptr) {
            *out_ptr = *a_ptr + *b_ptr;
        },
        [](T* out_ptr, const Vectorized<T> a_vec, const Vectorized<T> b_vec) {
            Vectorized<T> out_vec = a_vec + b_vec;
            out_vec.store(out_ptr);
        });
        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::subtract(const Tensor<T>& a, const Tensor<T>& b) const {
        auto result = std::make_shared<CPUTensor<T>>();
        auto iter = TensorIterator<T>::binary_op(*result, a, b);

        iter.for_each([](T** data, size_t size) {
            T* destination = data[0];
            const T* a_ptr = data[1];
            const T* b_ptr = data[2];

            if constexpr (std::is_same_v<T, float>) {
                cblas_scopy(size, a_ptr, 1, destination, 1);
                cblas_saxpy(size, -1.0f, b_ptr, 1, destination, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dcopy(size, a_ptr, 1, destination, 1);
                cblas_daxpy(size, -1.0, b_ptr, 1, destination, 1);
            } else {
                for (size_t i = 0; i < size; ++i) {
                    destination[i] = a_ptr[i] - b_ptr[i];
                }
            }
        });

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::multiply(const Tensor<T>& a, const Tensor<T>& b) const {
        return nullptr;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::divide(const Tensor<T>& a, const Tensor<T>& b) const {
        return nullptr;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) const {
        // Get shapes of tensors a and b
        const std::vector<index_t> a_shape = a.get_shape().dims();
        const std::vector<index_t> b_shape = b.get_shape().dims();

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
            Utils::calc_broadcast_shape(a_shape, b_shape, true);

        // Allocate result tensor
        auto result = std::make_shared<CPUTensor<T>>(Shape{result_shape});

        // Compute the strides for each tensor
        const std::vector<index_t> result_strides = result->get_strides();

        const index_t depth = static_cast<index_t>(result_shape.size()) - 2;
        index_t m = a_shape[a_shape.size() - 2];
        index_t k = a_shape[a_shape.size() - 1];
        index_t n = b_shape[b_shape.size() - 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Calculate the number of 2D matrices
        index_t num_matrices = 1;
        for (index_t i = 0; i < depth; ++i) {
            num_matrices *= result_shape[i];
        }

        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result->mutable_data();

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

            const T* a_sub = a_data + a_offset;
            const T* b_sub = b_data + b_offset;
            T* result_sub = result_data + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else {
                // Fallback scalar implementation
                for (index_t i = 0; i < m; ++i) {
                    for (index_t j = 0; j < n; ++j) {
                        T sum = 0;
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

    // Explicit template instantiation
    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
};


