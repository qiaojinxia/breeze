#include <iostream>
#include <armadillo>
#include "node.h"
#include "CPUTensor.h"
#include <chrono>
// 定义宏来测量代码段的运行时间
#define MEASURE_TIME(code_to_measure) do { \
auto start = std::chrono::high_resolution_clock::now(); \
code_to_measure; \
auto end = std::chrono::high_resolution_clock::now(); \
std::chrono::duration<double, std::milli> elapsed = end - start; \
std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl; \
} while (0)

using namespace Breeze;
int main() {
    // 创建节点并初始化值为矩阵
    arma::mat x_value = arma::ones<arma::mat>(2, 2) * 2;

    const auto x = std::make_shared<Node>(x_value);

    auto y = *x * x;


    auto z = *y + 3;


    auto w = *z * 2;


    Node::backward(w);

    const CPUTensor<float> tensor1({3, 2, 7, 1});
    tensor1.fill(2.0);

    const CPUTensor<float> tensor2({3, 2, 1, 7});
    tensor2.fill(3.0);


    const CPUTensor<float> tensor3({2, 3, 4});
    tensor3.fill(4.0);

    const CPUTensor<float> tensor4({2, 3, 4});
    tensor4.fill(2.0);


    const auto r1 = tensor1.matmul(tensor2);
    std::cout << *r1 << std::endl;


    const auto r2 = tensor3 + tensor4;
    std::cout << *r2 << std::endl;


    const auto r3 = tensor3 * tensor4;
    std::cout << *r3 << std::endl;


    CPUTensor<float> tensor5({1, 3, 4});
    tensor5.fill(4.0);

    CPUTensor<float> tensor6({2, 1, 4});
    tensor6.fill(2.0);


    tensor5.broadcast(tensor6);

    auto t1 = tensor5 + tensor6;

    std::cout << *t1 << std::endl;

    // MEASURE_TIME(const auto r1 = tensor3 * tensor4; );
    // MEASURE_TIME(const auto r2 = tensor3 + tensor4; );
    // MEASURE_TIME(const auto r4 = tensor3 / tensor4; );

    // 输出梯度
    std::cout << "∂L/∂x = " << x->grad << std::endl;
    std::cout << "∂L/∂y = " << y->grad << std::endl;

    return 0;
}
