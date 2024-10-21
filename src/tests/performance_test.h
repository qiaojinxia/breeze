//
// Created by caomaobay on 2024/10/20.
//

#ifndef PERFORMANCE_TEST_H
#define PERFORMANCE_TEST_H
#include "CPUTensor.h"
#include <iostream>
#include <vector>
#include <armadillo>
using namespace Breeze;
#include "../CPUTensor.h"
#include "../common/Macro.h"
static void test_Performance() {
    // 测试 std
    {

        const auto a = Tensor<float>::create_tensor({10000},1);
        MEASURE_TIME({
            const auto b = a->std({0});
            // std::cout << "按列计算的标准差（应该为 0）:\n" << *b << std::endl;
        });

        const arma::mat A = arma::ones<arma::mat>(10000);
        MEASURE_TIME({
            // 按列计算标准差
            arma::vec std_dev_col = arma::stddev(A, 0, 0); // 计算每列的标准差
            std::cout << "按列计算的标准差（应该为 0）:\n" << std_dev_col << std::endl;
      });
    }

    {
        //   const auto a = Tensor<float>::create_tensor({1000,1000},1);
        //   const auto b = Tensor<float>::create_tensor({1000,1000},1);
        //   MEASURE_TIME({
        //       const auto c = *a + *b;
        //   });
        //
        //   arma::mat A = arma::ones<arma::mat>(1000, 1000);
        //   arma::mat B = arma::ones<arma::mat>(1000, 1000);
        //   MEASURE_TIME({
        //       // 按列计算标准差
        //        arma::mat C = A + B;
        //
        // });

    }
};



#endif //PERFORMANCE_TEST_H
