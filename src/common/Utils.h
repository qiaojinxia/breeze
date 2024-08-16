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
        std::vector<size_t> strides(shape.size(), 1);
        const auto s_size = static_cast<int>(shape.size());
        for (int i = s_size - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
};

#endif //UTILS_H
