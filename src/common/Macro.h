//
// Created by mac on 2024/8/10.
//

#ifndef MACRO_H
#define MACRO_H
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

#endif //MACRO_H
