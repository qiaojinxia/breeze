#include "tests/test_cases.h"
#include <omp.h>
using namespace Breeze;
int main() {
    const int size = 1000000;
    double a[size];

    // 初始化数组
    for (int i = 0; i < size; i++) {
        a[i] = 1.0;
    }

    double sum = 0.0;

    // 获取开始时间
    double startTime = omp_get_wtime();
    // 使用 OpenMP 进行并行计算
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
#pragma omp master
        {
            std::cout << "实际使用的线程数：" << omp_get_num_threads() << std::endl;
        }
        sum += a[i];
    }

    // 获取结束时间
    double endTime = omp_get_wtime();

    std::cout << "总和: " << sum << std::endl;
    std::cout << "计算耗时: " << (endTime - startTime) << " 秒" << std::endl;

    // 打印出使用的线程数
    std::cout << "使用的线程数: " << omp_get_max_threads() << std::endl;

    TensorTest::run_all_tests();

    return 0;

}
