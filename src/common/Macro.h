//
// Created by mac on 2024/8/10.
//

#ifndef MACRO_H
#define MACRO_H
#include "Const.h"
// 定义宏来测量代码段的运行时间
#define MEASURE_TIME(code_to_measure) do { \
    auto start = std::chrono::high_resolution_clock::now(); \
    code_to_measure; \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> elapsed = end - start; \
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl; \
    } while (0)
#define KEEP {0, KEEP_ALL, 1}
#endif //MACRO_H
