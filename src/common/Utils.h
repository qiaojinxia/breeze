#ifndef UTILS_H
#define UTILS_H

#include <cassert>
#include <vector>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <tuple>
#include <iostream>
#include <cstdarg>
#include <valarray>

#include "Macro.h"
class Utils {
public:
    // 计算 strides 的函数
    static std::vector<index_t> compute_strides(const std::vector<index_t>& shape) {
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        if (shape.empty()) {
            return {};
        }
        std::vector<index_t> strides(shape.size(), 0);
        index_t accumulated_stride = 1;
        for (int32_t i = static_cast<int32_t>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = accumulated_stride;
            accumulated_stride *= shape[i];
        }
        return strides;
    }

    static std::vector<index_t> compute_strides_with_origin(const std::vector<index_t>& shape, const std::vector<index_t>& original_strides) {
        if (shape.empty()) {
            return {};
        }
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        if (shape.size() != original_strides.size()) {
            throw std::invalid_argument("Shape and original strides must have the same size.");
        }

        std::vector<index_t> new_strides(shape.size(), 0);
        index_t accumulated_stride = 1;

        for (int32_t i = static_cast<int32_t>(shape.size()) - 1; i >= 0; --i) {
            if (original_strides[i] != 0) {
                new_strides[i] = accumulated_stride;
                accumulated_stride *= shape[i];
            }
        }

        return new_strides;
    }

    static std::string Format(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);

        // Determine the required buffer size
        va_list args_copy;
        va_copy(args_copy, args);
        const int size = vsnprintf(nullptr, 0, fmt, args_copy);
        va_end(args_copy);

        // Allocate buffer and print the formatted string
        std::string buffer(size + 1, '\0');
        vsnprintf(&buffer[0], size + 1, fmt, args);

        va_end(args);
        return buffer;
    }

    static std::vector<int64_t> parseSliceString(const std::string& s, const int64_t dim_size) {
        std::vector<int64_t> result;
        std::istringstream iss(s);
        std::string token;

        while (std::getline(iss, token, ':')) {
            if (token.empty()) {
                // 对于空字段，我们插入一个特殊值
                result.push_back(std::numeric_limits<int32_t>::max());
            } else {
                result.push_back(std::stoi(token));
            }
        }

        // 填充默认值
        while (result.size() < 3) {
            result.push_back(std::numeric_limits<int32_t>::max());
        }

        // 处理步长
        if (result[2] == std::numeric_limits<int32_t>::max()) result[2] = 1;

        // 处理起始和结束索引
        if (result[2] > 0) {
            // 正步长
            if (result[0] == std::numeric_limits<int32_t>::max()) result[0] = 0;
            if (result[1] == std::numeric_limits<int32_t>::max()) result[1] = dim_size;
        } else if (result[2] < 0) {
            // 负步长
            if (result[0] == std::numeric_limits<int32_t>::max()) result[0] = dim_size - 1;
            if (result[1] == std::numeric_limits<int32_t>::max()) result[1] = -1;
        } else {
            // 步长为0，这是无效的
            throw std::invalid_argument("Slice step cannot be zero");
        }

        return result;
    }

    template<typename scalar_t>
    static void compare_tensor_data(const scalar_t* tensor_data, const std::vector<std::vector<scalar_t>>& expected, float epsilon = 1e-6) {
        index_t index = 0;
        for (index_t i = 0; i < static_cast<index_t>(expected.size()); ++i) {
            for (index_t j = 0; j < static_cast<index_t>(expected[i].size()); ++j) {
                if (std::abs(tensor_data[index] - expected[i][j]) > epsilon) {
                    std::ostringstream oss;
                    oss << "Mismatch at position [" << i << "][" << j << "]: "
                        << "Expected " << expected[i][j] << ", but got " << tensor_data[index];
                    throw std::runtime_error(oss.str());
                }
                ++index;
            }
        }
        std::cout << "Tensor data matches expected values." << std::endl;
    }

    static std::tuple<std::vector<index_t>, std::vector<index_t>, std::vector<index_t>>
    calc_matmul_shape(const std::vector<index_t>& shape1, const std::vector<index_t>& shape2) {
        // 计算目标形状
        std::vector<index_t> targetShape;
        const int32_t maxDims = static_cast<int32_t>(std::max(shape1.size(), shape2.size()));
        targetShape.resize(maxDims);

        // 处理除最后两个维度之外的部分
        for (int32_t i = 0; i < maxDims - 2; ++i) {
            const index_t dim1 = i < static_cast<int32_t>(shape1.size()) - 2 ? shape1[i] : 1;
            const index_t dim2 = i < static_cast<int32_t>(shape2.size()) - 2 ? shape2[i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::runtime_error("Incompatible shapes for broadcasting in matmul");
            }
            targetShape[i] = std::max(dim1, dim2);
        }

        // 处理最后两个维度
        targetShape[maxDims - 2] = shape1[shape1.size() - 2];
        targetShape[maxDims - 1] = shape2[shape2.size() - 1];

        // 计算输入向量的步长
        std::vector<index_t> strides1(maxDims, 0), strides2(maxDims, 0);
        index_t stride1 = 1, stride2 = 1;
        // 矩阵乘法的步长计算
        for (int32_t i = static_cast<int32_t>(shape1.size()) - 1; i >= 0; --i) {
            strides1[maxDims - shape1.size() + i] = shape1[i] == 1 ? 0 : stride1;
            stride1 *= shape1[i];
        }
        for (int32_t i = static_cast<int32_t>(shape2.size()) - 1; i >= 0; --i) {
            strides2[maxDims - shape2.size() + i] = shape2[i] == 1 ? 0 : stride2;
            stride2 *= shape2[i];
        }

        return {strides1, strides2, targetShape};
    }

    static std::vector<index_t> expand_strides(const std::vector<index_t>& input_shape,
        const std::vector<index_t>& output_shape, const std::vector<index_t>& strides) {
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
                    // 如果 原先形状为1 与原先不想等 处理广播情况
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

    [[nodiscard]] static index_t compute_offset(const std::vector<index_t>& counter, const std::vector<index_t>& strides_bytes) {
        index_t offset = 0;
        for (size_t i = 0; i < counter.size(); ++i) {
            offset += counter[i] * strides_bytes[i];
        }
        return offset;
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
};

#endif //UTILS_H