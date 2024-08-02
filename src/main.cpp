#include <iostream>
#include <armadillo>
#include "node.h"
#include "CPUTensor.h"

using namespace Breeze;
int main() {
    // 创建节点并初始化值为矩阵
    arma::mat x_value = arma::ones<arma::mat>(2, 2) * 2;

    const auto x = std::make_shared<Node>(x_value);

    auto y = *x * x;


    auto z = *y + 3;


    auto w = *z * 2;


    Node::backward(w);

    const CPUTensor<float> tensor1({2, 3, 7, 4, 5});
    tensor1.fill(2.0);

    const CPUTensor<float> tensor2({2, 3, 7, 5, 6});
    tensor2.fill(3.0);

    const auto tensor3 = tensor1 * tensor2;
    std::cout << *tensor3 << std::endl;

    // 输出梯度
    std::cout << "∂L/∂x = " << x->grad << std::endl;
    std::cout << "∂L/∂y = " << y->grad << std::endl;

    return 0;
}
