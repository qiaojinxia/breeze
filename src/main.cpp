#include "node.h"
#include <iostream>


int main() {

    auto const x = std::make_shared<MyBlob::Node>(1.0);

    auto const y = *x * x;
    auto const z = *y*y;

    MyBlob::Node::backward(z);

    // 输出梯度
    std::cout << "∂z/∂x = " << x->grad << std::endl;

    // (x * y) ** 2 对 x 求导 ∂z/∂x = ∂y/∂x * ∂z/∂y  = 2x  * 2y = (2 * 1) * (2 * 1) = 4
    return 0;
}
