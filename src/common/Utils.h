#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <limits>
#include <stdexcept>

class Utils {
public:
    // 计算 strides 的函数
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape vector cannot be empty.");
        }
        if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Shape size exceeds the maximum value that can be handled.");
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
            throw std::invalid_argument("Shape vector cannot be empty.");
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

};

#endif //UTILS_H
