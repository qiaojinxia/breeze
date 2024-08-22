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

        // 处理默认值
        if (result[0] == std::numeric_limits<int32_t>::max()) result[0] = 0;
        if (result[1] == std::numeric_limits<int32_t>::max()) result[1] = dim_size;
        if (result[2] == std::numeric_limits<int32_t>::max()) result[2] = 1;

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

};

#endif //UTILS_H
