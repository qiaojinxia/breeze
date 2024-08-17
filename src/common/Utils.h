#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <limits>
#include <stdexcept>

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

    static std::vector<size_t> compute_strides_with_zeros(const std::vector<size_t>& shape, const std::vector<size_t>& original_strides) {
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


};

#endif //UTILS_H
