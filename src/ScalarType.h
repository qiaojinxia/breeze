//
// Created by mac on 2024/9/12.
//

#ifndef scalar_t_H
#define scalar_t_H
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "./common/Macro.h"

namespace Breeze {

    enum class ScalarType {
        Float,
        Double,
        Bool,
    };

    template <typename ScalarType>
    struct TypeToScalarType;

    template <> struct TypeToScalarType<float> { static constexpr auto value = ScalarType::Float; };
    template <> struct TypeToScalarType<double> { static constexpr auto value = ScalarType::Double; };
    template <> struct TypeToScalarType<bool> { static constexpr auto value = ScalarType::Bool; };

    inline std::vector<index_t> calc_strides_bytes(const ScalarType scalar_t, const std::vector<index_t>& strides) {
        index_t byte_size;
        switch (scalar_t) {
            case ScalarType::Float:
                byte_size = sizeof(float);
            break;
            case ScalarType::Double:
                byte_size = sizeof(double);  // 注意：这里应该是 double，而不是 float
            break;
            case ScalarType::Bool:
                byte_size = sizeof(bool);    // 使用 bool 而不是 float
            break;
            // 可以根据需要添加更多的标量类型
            default:
                throw std::runtime_error("Unsupported ScalarType");
        }

        std::vector<index_t> byte_strides;
        byte_strides.reserve(strides.size());

        for (const auto& stride : strides) {
            byte_strides.push_back(stride * byte_size);
        }

        return byte_strides;
    }

    template<typename ScalarT1, typename ScalarT2>
    struct BinaryOpResultType {
        using type = std::common_type_t<ScalarT1, ScalarT2>;
    };

    // Specializations for specific type combinations
    template<>
    struct BinaryOpResultType<float, double> {
        using type = double;
    };

    template<>
  struct BinaryOpResultType<float, float> {
        using type = float;
    };

    template<>
  struct BinaryOpResultType<double, double> {
        using type = double;
    };

    template<>
    struct BinaryOpResultType<double, float> {
        using type = double;
    };

}
#endif //scalar_t_H
