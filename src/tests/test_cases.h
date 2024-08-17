//
// Created by caomaobay on 2024/8/17.
//

#ifndef TEST_SQUEEZE_H
#define TEST_SQUEEZE_H
#include <iostream>
#include <vector>
#include <cassert>
#include "../CPUTensor.h"
#include "../common/Macro.h"
#include <cassert>

using namespace Breeze;
class TensorTest {
public:
    // 打印张量形状
    static void print_shape(const std::vector<size_t>& shape) {
        std::cout << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")";
    }

    // 测试 expand 操作
    static void test_expand() {
        // 0维度（标量）进行 expand
        CPUTensor<float> scalar({}); // 假设标量初始化
        scalar.fill(1.0);
        const auto expanded_scalar = scalar.expand({2,-1, 2});

        assert(expanded_scalar->get_shape().dims() == std::vector<size_t>({2, 1 ,2}));
        std::cout << *expanded_scalar << std::endl;

        // 1维度扩展
        CPUTensor<float> tensor1d({3});
        tensor1d.fill(2.0);
        const auto expanded_1d = tensor1d.expand({3, 3});
        assert(expanded_1d->get_shape().dims() == std::vector<size_t>({3, 3}));
        std::cout << "Expanded 1D tensor shape: ";
        print_shape(expanded_1d->get_shape().dims());
        std::cout << std::endl;
        std::cout << *expanded_1d << std::endl;

        // 使用 -1 进行动态扩展
        CPUTensor<float> tensor2({1, 4, 1});
        tensor2.fill(1.0);
        const auto expanded_tensor2 = tensor2.expand({3, -1, 2});
        assert(expanded_tensor2->get_shape().dims() == std::vector<size_t>({3, 4, 2}));
        std::cout << "Expanded tensor2 shape with -1: ";
        print_shape(expanded_tensor2->get_shape().dims());
        std::cout << std::endl;
        std::cout << *expanded_tensor2 << std::endl;


        // 0维度（标量）进行 expand
        CPUTensor<float> scalar1({}); // 假设标量初始化
        scalar.fill(3.0);
        const auto expanded3_scalar = scalar.expand({5});

        assert(expanded3_scalar->get_shape().dims() == std::vector<size_t>({5}));
        std::cout << *expanded3_scalar << std::endl;

        // 错误情况：不兼容的形状
        try {
            const auto invalid_expand = tensor2.expand({2, 3, 4});
            assert(false && "Expected expand to throw an exception for incompatible shape.");
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught expected error for incompatible shape: " << e.what() << std::endl;
        }
    }

    // 测试 unsqueeze 操作
    static void test_unsqueeze() {
        // Basic unsqueeze
        CPUTensor<float> tensor1({2, 4, 3});
        const auto unsqueezed_tensor1 = tensor1.unsqueeze(1);
        assert(unsqueezed_tensor1->get_shape().dims() == std::vector<size_t>({2, 1, 4, 3}));
        std::cout << "Unsqueezed tensor1 shape: ";
        print_shape(unsqueezed_tensor1->get_shape().dims());
        std::cout << std::endl;

        // Edge case: Unsqueezing a 1D tensor
        CPUTensor<float> tensor2({4});
        const auto unsqueezed_tensor2 = tensor2.unsqueeze(0);
        assert(unsqueezed_tensor2->get_shape().dims() == std::vector<size_t>({1, 4}));
        std::cout << "Unsqueezed tensor2 shape: ";
        print_shape(unsqueezed_tensor2->get_shape().dims());
        std::cout << std::endl;

        // Edge case: Unsqueezing an already unsqueezed dimension
        const auto unsqueezed_tensor3 = unsqueezed_tensor2->unsqueeze(2);
        assert(unsqueezed_tensor3->get_shape().dims() == std::vector<size_t>({1, 4, 1}));
        std::cout << "Unsqueezed tensor3 shape: ";
        print_shape(unsqueezed_tensor3->get_shape().dims());
        std::cout << std::endl;
    }

    // 测试 squeeze 操作
    static void test_squeeze() {
        // Basic squeeze
        CPUTensor<float> tensor1({2, 1, 4, 3});
        const auto squeezed_tensor1 = tensor1.squeeze(1);
        assert(squeezed_tensor1->get_shape().dims() == std::vector<size_t>({2, 4, 3}));
        std::cout << "Squeezed tensor1 shape: ";
        print_shape(squeezed_tensor1->get_shape().dims());
        std::cout << std::endl;

        // Edge case: Squeeze a tensor with no unit dimensions
        CPUTensor<float> tensor2({2, 4, 3});
        const auto squeezed_tensor2 = tensor2.squeeze(1); // Squeezing non-unit dimension
        assert(squeezed_tensor2->get_shape().dims() == tensor2.get_shape().dims());
        std::cout << "Squeezed tensor2 shape (no change expected): ";
        print_shape(squeezed_tensor2->get_shape().dims());
        std::cout << std::endl;

        // Edge case: Squeezing an already squeezed dimension
        CPUTensor<float> tensor3({1});
        const auto squeezed_tensor3 = tensor3.squeeze(0);
        assert(squeezed_tensor3->get_shape().dims() == std::vector<size_t>({}));
        std::cout << "Squeezed tensor3 shape (empty): ";
        print_shape(squeezed_tensor3->get_shape().dims());
        std::cout << std::endl;
    }

    // 测试 view 操作
    static void test_view() {
        // 基本 view 操作
        CPUTensor<float> tensor1({2, 4, 3});
        const auto viewed_tensor1 = tensor1.view({4, 6});
        assert(viewed_tensor1->get_shape().dims() == std::vector<size_t>({4, 6}));
        std::cout << "Viewed tensor1 shape: ";
        print_shape(viewed_tensor1->get_shape().dims());
        std::cout << std::endl;

        // 使用 -1 自动计算维度
        CPUTensor<float> tensor2({3, 4, 5});
        const auto viewed_tensor2 = tensor2.view({6, -1});
        assert(viewed_tensor2->get_shape().dims() == std::vector<size_t>({6, 10}));
        std::cout << "Viewed tensor2 shape: ";
        print_shape(viewed_tensor2->get_shape().dims());
        std::cout << std::endl;

        // 增加维度
        CPUTensor<float> tensor3({12});
        const auto viewed_tensor3 = tensor3.view({3, 2, 2});
        assert(viewed_tensor3->get_shape().dims() == std::vector<size_t>({3, 2, 2}));
        std::cout << "Viewed tensor3 shape: ";
        print_shape(viewed_tensor3->get_shape().dims());
        std::cout << std::endl;

        // 减少维度
        CPUTensor<float> tensor4({2, 3, 4});
        const auto viewed_tensor4 = tensor4.view({24});
        assert(viewed_tensor4->get_shape().dims() == std::vector<size_t>({24}));
        std::cout << "Viewed tensor4 shape: ";
        print_shape(viewed_tensor4->get_shape().dims());
        std::cout << std::endl;

        // 保持相同的元素总数，但改变形状
        CPUTensor<float> tensor5({3, 8});
        const auto viewed_tensor5 = tensor5.view({4, 2, 3});
        assert(viewed_tensor5->get_shape().dims() == std::vector<size_t>({4, 2, 3}));
        std::cout << "Viewed tensor5 shape: ";
        print_shape(viewed_tensor5->get_shape().dims());
        std::cout << std::endl;

        // 测试错误情况：元素总数不匹配
        CPUTensor<float> tensor6({2, 3, 4});
        try {
            auto viewed_tensor6 = tensor6.view({5, 5});
            std::cout << "This should not be printed." << std::endl;
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught expected exception: " << e.what() << std::endl;
        }
    }


    // 测试 reshape 操作
    static void test_reshape() {
        // 创建一个张量并填充数据
        CPUTensor<float> tensor1({2, 3, 4});
        tensor1.fill(1.0);

        // 打印原始张量
        std::cout << "Original tensor1:" << std::endl;
        std::cout << tensor1 << std::endl;

        // 重新定义张量的形状
        const auto reshaped_tensor1 = tensor1.reshape({4, 3, 2});
        std::cout << "Reshaped tensor1 (4, 3, 2):" << std::endl;
        std::cout << *reshaped_tensor1 << std::endl;
        assert(reshaped_tensor1->get_shape().dims() == std::vector<size_t>({4, 3, 2}));

        // 动态维度的重塑
        const auto reshaped_tensor2 = tensor1.reshape({-1, 6});
        std::cout << "Reshaped tensor1 (-1, 6):" << std::endl;
        std::cout << *reshaped_tensor2 << std::endl;
        assert(reshaped_tensor2->get_shape().dims() == std::vector<size_t>({4, 6}));

        // 另一种动态维度的重塑
        const auto reshaped_tensor3 = tensor1.reshape({6, -1});
        std::cout << "Reshaped tensor1 (6, -1):" << std::endl;
        std::cout << *reshaped_tensor3 << std::endl;
        assert(reshaped_tensor3->get_shape().dims() == std::vector<size_t>({6, 4}));

        // 错误情况：不兼容的形状
        try {
            const auto reshaped_tensor4 = tensor1.reshape({3, 3, 3});
            assert(false && "Expected unsqueeze to throw an exception for out-of-bounds axis.");
        } catch (const std::invalid_argument& e) {
            std::cout << "Reshaped tensor1 (3, 3, 3):" << std::endl;
        }

        // 非连续张量的重塑
        const auto tensor2 = tensor1.slice({S_(0, 1), KEEP, KEEP});
        tensor2->fill(2.0);
        std::cout << "Non-contiguous tensor2:" << std::endl;
        std::cout << *tensor2 << std::endl;

        // 重新定义非连续张量的形状
        const auto reshaped_tensor5 = tensor2->reshape({2, -1});
        std::cout << "Reshaped tensor2 (2, -1):" << std::endl;
        std::cout << *reshaped_tensor5 << std::endl;
        assert(reshaped_tensor5->get_shape().dims() == std::vector<size_t>({2, 6}));


        CPUTensor<float> tensor6({});
        tensor6.fill(6.0);
        // 另一种动态维度的重塑
        const auto reshaped_tensor6 = tensor6.reshape({1, 1, -1});
        std::cout << "Reshaped tensor6 (1, 1, 1):" << std::endl;
        std::cout << *reshaped_tensor6 << std::endl;
        assert(reshaped_tensor6->get_shape().dims() == std::vector<size_t>({1, 1, 1}));

    }

    // 测试非连续张量的操作
    static void test_non_contiguous() {
        CPUTensor<float> tensor2({2, 1});
        tensor2.fill(2.0);
        const auto tensor2_ex = tensor2.expand({2, 3});
        assert(tensor2_ex->get_shape().dims() == std::vector<size_t>({2, 3}));
        std::cout << "Expanded tensor2 shape: ";
        print_shape(tensor2.get_shape().dims());
        std::cout << std::endl;

        const auto unsqueezed_tensor2 = tensor2.unsqueeze(0);
        assert(unsqueezed_tensor2->get_shape().dims() == std::vector<size_t>({1, 2, 3}));
        std::cout << "Unsqueezed tensor2 shape: ";
        print_shape(unsqueezed_tensor2->get_shape().dims());
        std::cout << std::endl;

        const auto squeezed_tensor2 = unsqueezed_tensor2->squeeze(0);
        assert(squeezed_tensor2->get_shape().dims() == std::vector<size_t>({2, 3}));
        std::cout << "Squeezed tensor2 shape: ";
        print_shape(squeezed_tensor2->get_shape().dims());
        std::cout << std::endl;
    }

    static void test_cat() {
        // 0维度标量拼接（应报错）
        try {
            CPUTensor<float> scalar1({});
            scalar1.fill(2.0);
            CPUTensor<float> scalar2({});
            scalar2.fill(1.0);
            const auto cat_scalar = CPUTensor<float>::cat({&scalar1, &scalar2}, 0);
            assert(false && "Expected cat to throw an exception for scalars.");
        } catch (const std::invalid_argument& e) {
            std::cout << "Success test case: "  << std::endl;
        }

        // 1维度拼接
        CPUTensor<float> tensor1d_1({3});
        tensor1d_1.fill(2.0);
        CPUTensor<float> tensor1d_2({3});
        tensor1d_2.fill(1.0);
        const auto cat_1d = CPUTensor<float>::cat({&tensor1d_1, &tensor1d_2}, 0);
        assert(cat_1d->get_shape().dims() == std::vector<size_t>({6}));
        std::cout << "1D tensor cat result: ";
        print_shape(cat_1d->get_shape().dims());
        std::cout << std::endl;

        // n维度连续拼接
        CPUTensor<float> tensor2d_1({2, 2});
        tensor2d_1.fill(3.0);
        CPUTensor<float> tensor2d_2({2, 2});
        tensor2d_2.fill(4.0);
        const auto cat_2d = CPUTensor<float>::cat({&tensor2d_1, &tensor2d_2}, 0);
        assert(cat_2d->get_shape().dims() == std::vector<size_t>({4, 2}));
        std::cout << "2D tensor cat result: ";
        print_shape(cat_2d->get_shape().dims());
        std::cout << std::endl;

        // n维度非连续拼接
        CPUTensor<float> tensor3d_1({2, 3, 4});
        tensor3d_1.fill(1.0);

        CPUTensor<float> tensor3d_2({2, 3, 4});
        tensor3d_2.fill(2.0);

        // Slice to create non-contiguous tensors
        const auto sliced_tensor1 = tensor3d_1.slice({S_(0, 2), S_(1, 3), KEEP});
        const auto sliced_tensor2 = tensor3d_2.slice({S_(0, 2), S_(1, 3), KEEP});

        // Concatenate along a specific axis
        const auto cat_non_contiguous = CPUTensor<float>::cat({sliced_tensor1.get(), sliced_tensor2.get()}, 1);

        // Check the shape
        assert(cat_non_contiguous->get_shape().dims() == std::vector<size_t>({2, 4, 4}));
        std::cout << "High-dimensional non-contiguous tensor cat result: ";
        print_shape(cat_non_contiguous->get_shape().dims());
        std::cout << std::endl;
        std::cout << *cat_non_contiguous << std::endl;


        // Create a 1D tensor
        CPUTensor<float> tensor1d({6});
        for (int i = 0; i < 6; ++i) {
            tensor1d.set_value({static_cast<size_t>(i)}, static_cast<float>(i));
        }
        // Slice to create non-contiguous tensors (e.g., take every other element)
        const auto sliced_tensor3 = tensor1d.slice({S3_(0, 5, -2)}); // [0, 2]
        const auto sliced_tensor4 = tensor1d.slice({S3_(1, 6, -2)}); // [1, 3]

        std::cout << *sliced_tensor3 << std::endl;
        std::cout << *sliced_tensor4 << std::endl;

        // Concatenate the slices
        const auto cat_non_contiguous_1d = CPUTensor<float>::cat({sliced_tensor3.get(), sliced_tensor4.get()}, 0);

        // Check the shape
        assert(cat_non_contiguous_1d->get_shape().dims() == std::vector<size_t>({6}));
        std::cout << "1D non-contiguous tensor cat result: ";
        print_shape(cat_non_contiguous_1d->get_shape().dims());
        std::cout << std::endl;

        std::cout << *cat_non_contiguous_1d << std::endl;
    }


    // 运行所有测试
    static void run_all_tests() {
        test_expand();
        test_cat();
        test_unsqueeze();
        test_squeeze();
        test_view();
        test_reshape();
        // test_non_contiguous();
        std::cout << "All tests passed!" << std::endl;
    }
};




#endif //TEST_SQUEEZE_H
