#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

class Utils {
public:
    // 计算 strides 的函数
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        if (shape.empty()) {
            return {};
        }
        std::vector<size_t> strides(shape.size(), 0);
        size_t accumulated_stride = 1;
        for (int32_t i = static_cast<int32_t>(shape.size()) - 1; i >= 0; --i) {
                strides[i] = accumulated_stride;
                accumulated_stride *= shape[i];
        }
        return strides;
    }

    static std::vector<size_t> compute_strides_with_origin(const std::vector<size_t>& shape, const std::vector<size_t>& original_strides) {
        if (shape.empty()) {
            return{};
        }
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
        }
        if (shape.size() != original_strides.size()) {
            throw std::invalid_argument("Shape and original strides must have the same size.");
        }

        std::vector<size_t> new_strides(shape.size(), 0);
        size_t accumulated_stride = 1;

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

    static std::vector<int32_t> parseSliceString(const std::string& s, const int32_t dim_size) {
        std::vector<int32_t> result;
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

    template<typename T>
    static void compare_tensor_data(const T* tensor_data, const std::vector<std::vector<T>>& expected, float epsilon = 1e-6) {
        size_t index = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            for (size_t j = 0; j < expected[i].size(); ++j) {
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



    static std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
    calc_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2,const bool matmul) {
        // 计算目标形状
        std::vector<size_t> targetShape;
        const int maxDims = static_cast<int>(std::max(shape1.size(), shape2.size()));
        targetShape.resize(maxDims);
        if (matmul) {
            // 特殊处理矩阵乘法的情况
            if (shape1.size() < 2 || shape2.size() < 2) {
                throw std::runtime_error("For matmul, both shapes must have at least 2 dimensions");
            }
            // 处理除最后两个维度之外的部分
            for (int i = 0; i < maxDims - 2; ++i) {
                const size_t dim1 = i < shape1.size() - 2 ? shape1[i] : 1;
                const size_t dim2 = i < shape2.size() - 2 ? shape2[i] : 1;
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw std::runtime_error("Incompatible shapes for broadcasting in matmul");
                }
                targetShape[i] = std::max(dim1, dim2);
            }

            // 处理最后两个维度
            targetShape[maxDims - 2] = shape1[shape1.size() - 2];
            targetShape[maxDims - 1] = shape2[shape2.size() - 1];
        } else {
            for (int i = 0; i < maxDims; ++i) {
                const size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
                const size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    throw std::runtime_error("Incompatible shapes for broadcasting");
                }
                targetShape[maxDims - 1 - i] = std::max(dim1, dim2);
            }
        }

        // 计算输入向量的步长
        std::vector<size_t> strides1(maxDims, 0), strides2(maxDims, 0);
        int32_t stride1 = 1, stride2 = 1;
        // 矩阵乘法的步长计算
        for (int i = static_cast<int>(shape1.size()) - 1; i >= 0; --i) {
            strides1[maxDims - shape1.size() + i] = shape1[i] == 1 ? 0 : stride1;
            stride1 *= static_cast<int32_t>(shape1[i]);
        }
        for (int i = static_cast<int>(shape2.size()) - 1; i >= 0; --i) {
            strides2[maxDims - shape2.size() + i] = shape2[i] == 1 ? 0 : stride2;
            stride2 *= static_cast<int32_t>(shape2[i]);
        }

        return {strides1, strides2, targetShape};
    }

};

#endif //UTILS_H
