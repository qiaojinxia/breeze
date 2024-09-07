// Created by mac on 2024/7/31.
//
#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>
#include "TensorIterator.h"
#include "platform/SIMDFactory.h"

namespace Breeze {
    template<typename Dtype>
    void CPUTensorOps<Dtype>::fill(Tensor<Dtype>& a, Dtype value) const {
        TensorIterator<Dtype> iter;
        iter.add_output(a);
        iter.build();
        iter.cpu_kernel_vec(
        [&](Dtype* out_ptr) {
            *out_ptr = value;
        },
        [&](Dtype* out_ptr) {
            auto value_vec = Vectorized<Dtype>(value);
            value_vec.store(out_ptr);
        });
    }

    template<typename Dtype>
    std::shared_ptr<Tensor<Dtype>> CPUTensorOps<Dtype>::add(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const {
        auto result = std::make_shared<CPUTensor<Dtype>>();
        auto iter = TensorIterator<Dtype>::binary_op(*result, a, b);
        iter.cpu_kernel_vec(
        [](Dtype* out_ptr, const Dtype* a_ptr, const Dtype* b_ptr) {
            *out_ptr = *a_ptr + *b_ptr;
        },
        [](Dtype* out_ptr, const Vectorized<Dtype> a_vec, const Vectorized<Dtype> b_vec) {
            Vectorized<Dtype> out_vec = a_vec + b_vec;
            out_vec.store(out_ptr);
        });
        return result;
    }

    template<typename Dtype>
    std::shared_ptr<Tensor<Dtype>> CPUTensorOps<Dtype>::subtract(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const {
        auto result = std::make_shared<CPUTensor<Dtype>>();
        auto iter = TensorIterator<Dtype>::binary_op(*result, a, b);

        iter.for_each([](Dtype** data, size_t size) {
            Dtype* destination = data[0];
            const Dtype* a_ptr = data[1];
            const Dtype* b_ptr = data[2];

            if constexpr (std::is_same_v<Dtype, float>) {
                cblas_scopy(size, a_ptr, 1, destination, 1);
                cblas_saxpy(size, -1.0f, b_ptr, 1, destination, 1);
            } else if constexpr (std::is_same_v<Dtype, double>) {
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

    template<typename Dtype>
    std::shared_ptr<Tensor<Dtype>> CPUTensorOps<Dtype>::multiply(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const {
        return nullptr;
    }

    template<typename Dtype>
    std::shared_ptr<Tensor<Dtype>> CPUTensorOps<Dtype>::divide(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const {
        return nullptr;
    }

    template<typename Dtype>
    std::shared_ptr<Tensor<Dtype>> CPUTensorOps<Dtype>::matmul(const Tensor<Dtype>& a, const Tensor<Dtype>& b) const {
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
        auto result = std::make_shared<CPUTensor<Dtype>>(Shape{result_shape});

        // Compute the strides for each tensor
        const std::vector<index_t> result_strides = result->get_strides();

        const index_t depth = static_cast<index_t>(result_shape.size()) - 2;
        index_t m = a_shape[a_shape.size() - 2];
        index_t k = a_shape[a_shape.size() - 1];
        index_t n = b_shape[b_shape.size() - 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        auto alpha = static_cast<Dtype>(1.0);
        auto beta = static_cast<Dtype>(0.0);

        // Calculate the number of 2D matrices
        index_t num_matrices = 1;
        for (index_t i = 0; i < depth; ++i) {
            num_matrices *= result_shape[i];
        }

        const Dtype* a_data = a.data();
        const Dtype* b_data = b.data();
        Dtype* result_data = result->mutable_data();

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

            const Dtype* a_sub = a_data + a_offset;
            const Dtype* b_sub = b_data + b_offset;
            Dtype* result_sub = result_data + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<Dtype, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<Dtype, double>) {
                cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else {
                // Fallback scalar implementation
                for (index_t i = 0; i < m; ++i) {
                    for (index_t j = 0; j < n; ++j) {
                        Dtype sum = 0;
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


