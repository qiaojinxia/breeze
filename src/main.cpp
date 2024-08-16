#include <iostream>
#include <armadillo>
#include "node.h"
#include "CPUTensor.h"
#include "common/Macro.h"
#include <omp.h>
using namespace Breeze;
int main() {
    // 创建节点并初始化值为矩阵
    // arma::mat x_value = arma::ones<arma::mat>(2, 2) * 2;
    //
    // const auto x = std::make_shared<Node>(x_value);
    //
    // auto y = *x * x;
    //
    // auto z = *y + 3;
    //
    // auto w = *z * 2;
    //
    // Node::backward(w);

    CPUTensor<float> tensor1({3, 2, 7, 1});
    tensor1.fill(2.0);

    auto tensor1_s = tensor1.slice({S_(0, 1), KEEP, KEEP, KEEP});
    tensor1_s->fill(0.5);

    std::cout << tensor1 << std::endl;

    CPUTensor<float> tensor2({3, 2, 1, 7});
    tensor2.fill(1.7);
    // MEASURE_TIME(tensor1_s->matmul(tensor2));

    CPUTensor<float> tensor3({2, 1, 8});
    tensor3.fill(4.0);

    CPUTensor<float> tensor4({2, 3, 1});
    tensor4.fill(3.0);


    // MEASURE_TIME(tensor1_s->matmul(tensor2));
    const auto r1 = tensor1_s->matmul(tensor2);
    std::cout << *r1 << std::endl;

    const auto r2 = tensor3 + tensor4;
    std::cout << *r2 << std::endl;
    MEASURE_TIME(auto r21 = tensor3 * tensor4;);
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

    MEASURE_TIME(auto t2 = CPUTensor<float>::arrange(1,10,2));

    CPUTensor<float> tensor8(Shape{3, 2, 2, 6, 1});
    tensor8.fill(4.0);
    tensor8.expand(Shape{3, 2, 2, 6, 4});

    // 2 2 4 4
    auto s_tensor8 = tensor8.slice({KEEP,S_(1,2),S_(1,2),S_(2,6),S_(2,6)});
    s_tensor8->fill(1.5);

    CPUTensor<float> tensor9(Shape{3, 1, 1, 4, 2});
    tensor9.fill(2.0);
    CPUTensor<float> tensor10(Shape{3, 2, 1, 4, 2});
    tensor10.fill(3.0);

    std::cout << tensor8 << std::endl;
    std::cout << "tensor8 是否连续" << tensor8.is_contiguous()  << std::endl;
    std::cout << "切片 s_tensor8 是否连续" << s_tensor8->is_contiguous() << std::endl;
    //不连续拼接
    auto t4 = CPUTensor<float>::cat({dynamic_cast<CPUTensor<float>*>(s_tensor8.get()), &tensor9},2);
    //连续拼接
    MEASURE_TIME(auto t5 = CPUTensor<float>::cat({&tensor9, &tensor10},1));
    std::cout << *t4 << std::endl;

    auto s_tensor8_c = s_tensor8->clone();
    std::cout << *s_tensor8_c << std::endl;
    std::cout << "复制后是否连续" << s_tensor8_c->is_contiguous() << std::endl;

    // 切片 克隆后view
    auto s_tensor8_c_v = s_tensor8_c->view({2,-1});
    std::cout << *s_tensor8_c_v << std::endl;

    CPUTensor<float> tensor11(Shape{2, 12});
    tensor11.fill(2.0);

    // clone 后测试加法是否异常
    auto t5 = tensor11 + *s_tensor8_c_v;
    std::cout << *t5 << std::endl;

    // 对expand的进行 clone
    CPUTensor<float> tensor12(Shape{2, 1});
    tensor12.fill(6.0);
    tensor12.expand({2,12});
    auto t6 = tensor12.clone();
    std::cout << *t6 << std::endl;


    // CPUTensor<float> tensor13(Shape{12});
    tensor12.fill(7.0);
    auto t7 = tensor12.unsqueeze(0);
    std::cout << tensor12 << std::endl;
    std::cout << *t7 << std::endl;
    auto t8 = t7->squeeze(0);
    std::cout << *t8 << std::endl;


    CPUTensor<float> tensor13(Shape{1, 4});
    tensor13.fill(8.0);
    tensor13.expand({1,4});
    auto t9 = tensor13.squeeze(0);
    std::cout << *t9 << std::endl;

    // std::cout << *t5 << std::endl;
    // MEASURE_TIME(const auto r1 = tensor3 * tensor4;);
    // MEASURE_TIME(const auto r_2 = tensor3 + tensor4;);
    // MEASURE_TIME(const auto r4 = tensor3 / tensor4;);

    // 输出梯度
    // std::cout << "∂L/∂x = " << x->grad << std::endl;
    // std::cout << "∂L/∂y = " << y->grad << std::endl;

    return 0;
}
