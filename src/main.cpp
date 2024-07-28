#include <iostream>
#include <armadillo>
#include "node.h"
#include "loss_functions.h"
using namespace MyBlob;
int main() {
    // 创建节点并初始化值为矩阵
    arma::mat x_value = arma::ones<arma::mat>(2, 2) * 2;

    const auto x = std::make_shared<MyBlob::Node>(x_value);

    auto y = *x * x;


    auto z = *y + 3;


    auto w = *z * 2;


    MyBlob::Node::backward(w);


    // 输出梯度
    std::cout << "∂L/∂x = " << x->grad << std::endl;
    std::cout << "∂L/∂y = " << y->grad << std::endl;

    return 0;
}
