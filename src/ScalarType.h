//
// Created by mac on 2024/9/12.
//

#ifndef scalar_t_H
#define scalar_t_H
#include <stdexcept>
#include "./common/Macro.h"

namespace Breeze {

    enum class ScalarType {
        Float,
        Double,
        Bool,
        Int64,
        Int32,
    };

    template <typename ScalarType>
    struct TypeToScalarType;

    template <> struct TypeToScalarType<float> { static constexpr auto value = ScalarType::Float; };
    template <> struct TypeToScalarType<double> { static constexpr auto value = ScalarType::Double; };
    template <> struct TypeToScalarType<bool> { static constexpr auto value = ScalarType::Bool; };
    template <> struct TypeToScalarType<i64> { static constexpr auto value = ScalarType::Int64; };
    template <> struct TypeToScalarType<i32> { static constexpr auto value = ScalarType::Int32; };

    inline std::string scalar_type_to_string(const ScalarType type) {
        switch (type) {
            case ScalarType::Float:
                return "Float";
            case ScalarType::Double:
                return "Double";
            case ScalarType::Bool:
                return "Bool";
            case ScalarType::Int64:
                return "int64";
            case ScalarType::Int32:
                return "int32";
            default:
                return "Unknown";
        }
    }

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
                byte_size = sizeof(bool);
            break;
            case ScalarType::Int64:
                byte_size = sizeof(i64);
            break;
            case ScalarType::Int32:
                byte_size = sizeof(i32);
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

    // Type mapping entry
    template<typename T1, typename T2, typename Result>
    struct TypeMapping {
        using first_type = T1;
        using second_type = T2;
        using result_type = Result;
    };

    // Static type dictionary
    template<typename... Mappings>
    struct TypeDictionary {
        template<typename T1, typename T2>
        struct Lookup {
            using type = std::common_type_t<T1, T2>;  // default fallback
        };
    };

    // Specialization for non-empty dictionary
    template<typename Mapping, typename... Rest>
    struct TypeDictionary<Mapping, Rest...> {
        template<typename T1, typename T2>
        struct Lookup {
        private:
            using CurrentMatch = std::conjunction<
                std::is_same<T1, typename Mapping::first_type>,
                std::is_same<T2, typename Mapping::second_type>
            >;

            using NextLookup = typename TypeDictionary<Rest...>::template Lookup<T1, T2>;

        public:
            using type = std::conditional_t<
                CurrentMatch::value,
                typename Mapping::result_type,
                typename NextLookup::type
            >;
        };
    };

    // Define the type dictionary
    using BinaryOpTypeDict = TypeDictionary<
        TypeMapping<float, double, double>,
        TypeMapping<float, float, float>,
        TypeMapping<double, double, double>,
        TypeMapping<double, float, double>,
        TypeMapping<i64, i64, i64>,
        TypeMapping<i64, float, float>,
        TypeMapping<float, i64, float>,
        TypeMapping<i64, double, double>,
        TypeMapping<double, i64, double>
    >;

    // New simplified BinaryOpResultType
    template<typename ScalarT1, typename ScalarT2>
    struct BinaryOpResultType {
        using type = typename BinaryOpTypeDict::Lookup<ScalarT1, ScalarT2>::type;
    };

    // Helper alias template for easier usage
    template<typename T1, typename T2>
        using binary_op_result_t = typename BinaryOpResultType<T1, T2>::type;
    }

#endif //scalar_t_H

