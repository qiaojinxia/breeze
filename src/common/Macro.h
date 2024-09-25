//
// Created by mac on 2024/8/10.
//

#ifndef MACRO_H
#define MACRO_H

using index_t = int64_t;
#define DEFINE_BINARY_OP(OP_NAME, METHOD_NAME)                                  \
template<typename ScalarType>                                                  \
std::shared_ptr<TensorBase> CPUTensor<ScalarType>::OP_NAME(const TensorBase& rhs) const { \
    if (const auto* rhsFloat = dynamic_cast<const CPUTensor<float>*>(&rhs)) {   \
        return CPUTensorOps<ScalarType, float>::getInstance().METHOD_NAME(*this, *rhsFloat); \
    } else if (const auto* rhsDouble = dynamic_cast<const CPUTensor<double>*>(&rhs)) { \
        return CPUTensorOps<ScalarType, double>::getInstance().METHOD_NAME(*this, *rhsDouble); \
    } else {                                                                   \
        throw std::runtime_error(#OP_NAME " between tensors of different scalar types is not supported."); \
    }                                                                          \
}


#define DEFINE_BINARY_OP_VOID(OP_NAME, METHOD_NAME)                                  \
template<typename ScalarType>                                                  \
void CPUTensor<ScalarType>::OP_NAME(const TensorBase& rhs) { \
if (const auto* rhsFloat = dynamic_cast<const CPUTensor<float>*>(&rhs)) {   \
    CPUTensorOps<ScalarType, float>::getInstance().METHOD_NAME(*this, *rhsFloat); \
} else if (const auto* rhsDouble = dynamic_cast<const CPUTensor<double>*>(&rhs)) { \
    CPUTensorOps<ScalarType, double>::getInstance().METHOD_NAME(*this, *rhsDouble); \
} else {                                                                   \
throw std::runtime_error(#OP_NAME " between tensors of different scalar types is not supported."); \
}                                                                          \
}

#define DEFINE_UNARY_OP(OP_NAME) \
    template<typename ScalarType> \
    std::shared_ptr<TensorBase> CPUTensor<ScalarType>::OP_NAME() const { \
        return CPUTensorOps<ScalarType>::getInstance().OP_NAME(*this); \
    }


// 定义宏来测量代码段的运行时间
#define MEASURE_TIME(code_to_measure) do { \
    auto start = std::chrono::high_resolution_clock::now(); \
    code_to_measure; \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> elapsed = end - start; \
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl; \
    } while (0)

#define ASSERT_THROWS(expression, exception_type)                            \
    do {                                                                     \
        bool caught_expected_exception = false;                              \
    try {                                                                \
        expression;                                                      \
    } catch (const exception_type&) {                                    \
        caught_expected_exception = true;                                \
    } catch (...) {                                                      \
        std::cerr << "Caught unexpected exception type" << std::endl;    \
        assert(false && "Caught unexpected exception type");             \
    }                                                                    \
    if (!caught_expected_exception) {                                    \
        std::cerr << "Expected exception was not thrown" << std::endl;   \
        assert(false && "Expected exception was not thrown");            \
    }                                                                    \
} while (0)

typedef int8_t QInt8;

#define T_ERROR(msg) throw std::runtime_error(std::string("Error: ") + (msg));

#define BREEZE_ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #cond << "\n"; \
            oss << "Error at " << __FILE__ << ":" << __LINE__ << "\n"; \
            oss << "In function: " << __func__ << "\n"; \
            breeze_assert_print(oss, ##__VA_ARGS__); \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)


inline void breeze_assert_print(std::ostringstream& oss) {
    // Base case: do nothing
}

template<typename T, typename... Args>
void breeze_assert_print(std::ostringstream& oss, T&& first, Args&&... args) {
    oss << std::forward<T>(first);
    breeze_assert_print(oss, std::forward<Args>(args)...);
}

#ifdef NDEBUG
#define BREEZE_ASSERT_DEBUG(cond, ...) ((void)0)
#else
#define BREEZE_ASSERT_DEBUG(cond, ...) BREEZE_ASSERT(cond, ##__VA_ARGS__)
#endif


#define COMPARE_TENSOR_DATA(tensor_data, expected, epsilon) \
do { \
    size_t index = 0; \
    for (size_t i = 0; i < (expected).size(); ++i) { \
        for (size_t j = 0; j < (expected)[i].size(); ++j) { \
            if (std::abs((tensor_data)[index] - (expected)[i][j]) > (epsilon)) { \
                std::ostringstream oss; \
                oss << "Mismatch at position [" << i << "][" << j << "]: " \
                << "Expected " << (expected)[i][j] << ", but got " << (tensor_data)[index] \
                << " in " << __FILE__ << " at line " << __LINE__; \
                throw std::runtime_error(oss.str()); \
            } \
            ++index; \
        } \
    } \
    std::cout << "Tensor data matches expected values." << std::endl; \
} while(0)



// 专门用于 std::vector<size_t> 的 ASSERT_EQ 宏
#define ASSERT_EQ_VECTOR(actual, expected) \
    do { \
        const auto& a = (actual); \
        const auto& e = (expected); \
        if (a.size() != e.size()) { \
            std::ostringstream ss; \
            ss << "Assertion failed: " << #actual << " == " << #expected << "\n"; \
            ss << "Vector sizes do not match.\n"; \
            ss << "  Actual size: " << a.size() << "\n"; \
            ss << "Expected size: " << e.size() << "\n"; \
            ss << "    File: " << __FILE__ << "\n"; \
            ss << "    Line: " << __LINE__ << "\n"; \
            throw std::runtime_error(ss.str()); \
        } \
        for (size_t i = 0; i < a.size(); ++i) { \
            if (a[i] != e[i]) { \
                std::ostringstream ss; \
                ss << "Assertion failed: " << #actual << " == " << #expected << "\n"; \
                ss << "Vectors differ at index " << i << "\n"; \
                ss << "  Actual: " << a[i] << "\n"; \
                ss << "Expected: " << e[i] << "\n"; \
                ss << "    File: " << __FILE__ << "\n"; \
                ss << "    Line: " << __LINE__ << "\n"; \
                throw std::runtime_error(ss.str()); \
            } \
        } \
    } while (0)

#endif //MACRO_H
