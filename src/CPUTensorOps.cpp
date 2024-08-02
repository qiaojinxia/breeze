// Created by mac on 2024/7/31.
//

#include "CPUTensorOps.h"
#include "CPUTensor.h"
#include <cblas.h>

namespace Breeze {

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::add(const Tensor<T>& a, const Tensor<T>& b) const {
        // Verify that the dimensions of both tensors are compatible for addition
        if (a.get_shape() != b.get_shape()) {
            throw std::invalid_argument("Tensor shapes must be equal for addition.");
        }

        // Create a new tensor for the result with the same shape as the input tensors
        auto result = std::make_shared<CPUTensor<T>>(a.get_shape());

        // Number of elements (flattened size)
        const size_t num_elements = a.size();  // Assuming size() calculates the total number of elements

        // Using cblas_axpy for addition
        // y := a*x + y where a = 1, x = b.data(), y = result.data()
        T alpha = 1.0;
        if constexpr (std::is_same_v<T, float>) {
            cblas_saxpy(static_cast<int>(num_elements), alpha, b.data(), 1, result->data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_daxpy(static_cast<int>(num_elements), alpha, b.data(), 1, result->data(), 1);
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensorOps<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) const {
        // Get shapes of tensors a and b
        const std::vector<size_t>& a_shape = a.get_shape();
        const std::vector<size_t>& b_shape = b.get_shape();

        // Check for correct dimensions
        if (a_shape.size() < 2 || b_shape.size() < 2) {
            throw std::invalid_argument("Input tensors must have at least two dimensions for matrix multiplication.");
        }

        if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
            throw std::invalid_argument("The inner dimensions must match for matrix multiplication.");
        }

        // Determine the output shape
        std::vector<size_t> result_shape = a_shape;
        result_shape[result_shape.size() - 1] = b_shape[b_shape.size() - 1];

        // Allocate result tensor
        auto result = std::make_shared<CPUTensor<T>>(result_shape);

        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Compute the strides for each tensor
        const std::vector<size_t> a_strides = compute_strides(a_shape);
        const std::vector<size_t> b_strides = compute_strides(b_shape);
        const std::vector<size_t> result_strides = compute_strides(result_shape);

        // Call the recursive matrix multiplication
        multiply_non_recursive(a.data(), b.data(), result->data(), a_shape, b_shape, a_strides, b_strides, result_strides);

        return result;
    }

template<typename T>
void CPUTensorOps<T>::multiply_non_recursive(const T* a, const T* b, T* result, const std::vector<size_t>& a_shape,
                                             const std::vector<size_t>& b_shape,
                                             const std::vector<size_t>& a_strides, const std::vector<size_t>& b_strides,
                                             const std::vector<size_t>& result_strides) const {
        size_t depth = a_shape.size() - 2;
        size_t m = a_shape[depth];
        size_t k = a_shape[depth + 1];
        size_t n = b_shape[depth + 1];

        CBLAS_ORDER order = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        T alpha = static_cast<T>(1.0);
        T beta = static_cast<T>(0.0);

        // Calculate the number of 2D matrices (product of dimensions up to 'depth')
        size_t num_matrices = 1;
        for (size_t i = 0; i < depth; ++i) {
            num_matrices *= a_shape[i];
        }

        // Parallelize using OpenMP
        #pragma omp parallel for
        for (size_t idx = 0; idx < num_matrices; ++idx) {
            size_t a_offset = 0;
            size_t b_offset = 0;
            size_t result_offset = 0;
            size_t temp_idx = idx;

            // Calculate the offsets for each matrix slice
            for (size_t i = 0; i < depth; ++i) {
                const size_t index = temp_idx % a_shape[i];
                a_offset += index * a_strides[i];
                b_offset += index * b_strides[i];
                result_offset += index * result_strides[i];
                temp_idx /= a_shape[i];
            }

            const T* a_sub = a + a_offset;
            const T* b_sub = b + b_offset;
            T* result_sub = result + result_offset;

            // Use BLAS for matrix multiplication
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dgemm(order, transA, transB, m, n, k, alpha, a_sub, k, b_sub, n, beta, result_sub, n);
            }
        }
}

template<typename T>
[[nodiscard]] std::vector<size_t> CPUTensorOps<T>::compute_strides(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape vector cannot be empty.");
        }
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        std::vector<size_t> strides(shape.size(), 1);
        const auto s_size = static_cast<int>(shape.size());
        for (int i = s_size - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
}


    // Explicit template instantiation
    template class CPUTensorOps<float>;
    template class CPUTensorOps<double>;
};
