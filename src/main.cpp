#include <iostream>
#include <armadillo>
#include "node.h"
#include "CPUTensor.h"
#include "common/Macro.h"

using namespace Breeze;
int main() {
    // 创建节点并初始化值为矩阵
    arma::mat x_value = arma::ones<arma::mat>(2, 2) * 2;

    const auto x = std::make_shared<Node>(x_value);

    auto y = *x * x;


    auto z = *y + 3;


    auto w = *z * 2;


    Node::backward(w);

    CPUTensor<float> tensor1({3, 2, 7, 1});
    tensor1.fill(2.0);

    CPUTensor<float> tensor2({3, 2, 1, 7});
    tensor2.fill(3.0);


    CPUTensor<float> tensor3({2, 1, 8});
    tensor3.fill(4.0);

    CPUTensor<float> tensor4({2, 3, 1});
    tensor4.fill(3.0);

    // MEASURE_TIME(tensor1.matmul(tensor2););
    const auto r1 = tensor1.matmul(tensor2);
    std::cout << *r1 << std::endl;


    const auto r2 = tensor3 + tensor4;
    std::cout << *r2 << std::endl;


    const auto r3 = tensor3 * tensor4;

    auto r5 = *r2 + *r3;
    std::cout << *r5 << std::endl;

    auto rs = r3->slice({KEEP, KEEP, {0, 9, 2 }});
    std::cout << *rs << std::endl;
    auto rs1 = rs->slice({KEEP, KEEP, {0, 9, 2 }});
    std::cout << *rs1 << std::endl;


    CPUTensor<float> tensor5(Shape{1, 1});
    tensor5.fill(4.0);

    CPUTensor<float> tensor6(Shape{2, 1, 4});
    tensor6.fill(2.0);


    // auto t1 = tensor5 + tensor6;

    CPUTensor<float> tensor7(Shape{1, 4});
    tensor7.fill(1.0);
    tensor7.expand({3,4});
    std::cout << tensor7 << std::endl;

    // MEASURE_TIME(const auto r1 = tensor3 * tensor4; );
    // MEASURE_TIME(const auto r_2 = tensor3 + tensor4; );
    // MEASURE_TIME(const auto r4 = tensor3 / tensor4; );

    // 输出梯度
    std::cout << "∂L/∂x = " << x->grad << std::endl;
    std::cout << "∂L/∂y = " << y->grad << std::endl;

    return 0;
}
