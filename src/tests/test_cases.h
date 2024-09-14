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

using namespace Breeze;
class TensorTest {
public:

    // 打印张量形状
    static void print_shape(const std::vector<index_t>& shape) {
        std::cout << "(";
        for (index_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")";
    }

    // 测试 expand 操作
    static void test_expand() {
        {
            const auto t1 = CPUTensor<float>::scalar(4.0);
            const auto t2 = t1->expand({3});
            assert(t2->get_shape().dims() == std::vector<index_t>({3}));
            assert(t2->is_contiguous() == false);
            std::vector<std::vector<float>> expected = {
                { 4, 4 , 4 }};
            COMPARE_TENSOR_DATA(t2->contiguous()->data(), expected, 1e-6);
        }
        {
            const auto t1 = CPUTensor<float>::scalar(3.0);
            const auto t2 = t1->expand({3, 4});
            assert(t2->is_contiguous() == false);
            assert(t2->get_shape().dims() == std::vector<index_t>({3,  4}));
            std::vector<std::vector<float>> expected = {
                { 3, 3 , 3, 3 },
                { 3, 3 , 3, 3 },
                { 3, 3 , 3, 3 },
            };
            COMPARE_TENSOR_DATA(t2->contiguous()->data(), expected, 1e-6);
        }

        {
            const auto t1 = CPUTensor<float>::scalar(3.0);
            const auto t2 = t1->expand({0});
            assert(t2->get_shape().dims() == std::vector<index_t>({0}));
            assert(t2->is_contiguous() == true);
        }
        {
            const CPUTensor<float> t1({1, 2, 1},2.0);
            const auto t2 = t1.expand({2,2,1});
            assert(t2->get_shape().dims() == std::vector<index_t>({2,2,1}));
            assert(t2->is_contiguous() == false);
            assert(t2->data()[0] == 2.0f && t2->data()[1] == 2.0f );
        }

        {
            const CPUTensor<float> t1({1, 2, 1},2.0);
            const auto t2 = t1.expand({1,2,2});
            ASSERT_EQ_VECTOR(t2->get_shape().dims() , std::vector<index_t>({1,2,2}));
            assert(t2->is_contiguous() == false);
            assert(t2->data()[0] == 2.0f && t2->data()[1] == 2.0f  );
        }
        {
            const CPUTensor<float> t1({2, 1, 2},2.0);
            const auto t2 = t1.expand({2,4,2});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(), std::vector<index_t>({2,4,2}));
            assert(t2->is_contiguous() == false);
            const auto t3 = t2->contiguous();
            assert(t3->is_contiguous() == true);
            std::vector<std::vector<float>> expected = {
                { 2, 2 , 2, 2 },
                { 2, 2 , 2, 2 },
                { 2, 2 , 2, 2 },
                {2, 2 , 2, 2 }
            };
            COMPARE_TENSOR_DATA(t3->data(), expected, 1e-6);
        }

        {
            auto t1 = CPUTensor<float>::scalar(1);
            auto t2 = t1->expand({2, 4, 3}); // 维度数量不匹配
            std::vector<std::vector<float>> expected = {
                { 1, 1 , 1, 1, 1 , 1 },
                { 1, 1 , 1, 1, 1 , 1 },
                { 1, 1 , 1, 1, 1 , 1 },
                { 1, 1 , 1, 1, 1 , 1 }
            };
            assert(t2->is_contiguous() == false);
            ASSERT_EQ_VECTOR(t2->get_shape().dims(), std::vector<index_t>({2,4,3}));
            COMPARE_TENSOR_DATA(t2->contiguous()->data(), expected, 1e-6);
        }

        ASSERT_THROWS({
            CPUTensor<float> t1({2, 3});
            for (index_t i = 0;i<6;i++) {
                t1.mutable_data()[i] = i + 1;
            }
            auto _ = t1.expand({1, 3}); // 尝试减少非单一维度的大小
        }, std::invalid_argument);

        ASSERT_THROWS({
            CPUTensor<float> t1({2, 3});
            for (index_t i = 0;i<6; i++) {
              t1.mutable_data()[i] = i + 1;
            }
            auto _ = t1.expand({2, 3, 1}); // 新增维度
        }, std::invalid_argument);

    }

    // 测试 unsqueeze 操作
      static void test_unsqueeze() {
        // 辅助函数，用于检查 unsqueeze 操作的结果
        auto check_unsqueeze = [](const CPUTensor<float>& tensor, int dim, const std::vector<index_t>& expected_shape) {
            auto unsqueezed = tensor.unsqueeze(dim);
            assert(unsqueezed->get_shape().dims() == expected_shape);
            std::cout << "Unsqueezed tensor shape (dim " << dim << "): ";
            print_shape(unsqueezed->get_shape().dims());
            std::cout << std::endl;

            // 检查连续性
            assert(unsqueezed->is_contiguous());

            // 检查数据一致性
            assert(std::equal(tensor.data(), tensor.data() + tensor.num_elements(), unsqueezed->data()));
        };

        // 测试 1D 张量
        {
            const CPUTensor<float> tensor({4});
            check_unsqueeze(tensor, 0, {1, 4});
            check_unsqueeze(tensor, 1, {4, 1});
            check_unsqueeze(tensor, -1, {4, 1});
            check_unsqueeze(tensor, -2, {1, 4});
        }

        // 测试 2D 张量
        {
            const CPUTensor<float> tensor({2, 3});
            check_unsqueeze(tensor, 0, {1, 2, 3});
            check_unsqueeze(tensor, 1, {2, 1, 3});
            check_unsqueeze(tensor, 2, {2, 3, 1});
            check_unsqueeze(tensor, -1, {2, 3, 1});
            check_unsqueeze(tensor, -2, {2, 1, 3});
            check_unsqueeze(tensor, -3, {1, 2, 3});
        }

        // 测试 3D 张量
        {
            const CPUTensor<float> tensor({2, 3, 4});
            check_unsqueeze(tensor, 0, {1, 2, 3, 4});
            check_unsqueeze(tensor, 1, {2, 1, 3, 4});
            check_unsqueeze(tensor, 2, {2, 3, 1, 4});
            check_unsqueeze(tensor, 3, {2, 3, 4, 1});
            check_unsqueeze(tensor, -1, {2, 3, 4, 1});
            check_unsqueeze(tensor, -2, {2, 3, 1, 4});
            check_unsqueeze(tensor, -3, {2, 1, 3, 4});
            check_unsqueeze(tensor, -4, {1, 2, 3, 4});
        }

        // 测试 0D 张量 (标量)
        {
            const CPUTensor<float> tensor({});
            check_unsqueeze(tensor, 0, {1});
            check_unsqueeze(tensor, -1, {1});
        }

        // 测试边界情况：包含 1 的维度
        {
            const CPUTensor<float> tensor({2, 1, 3});
            check_unsqueeze(tensor, 1, {2, 1, 1, 3});
            check_unsqueeze(tensor, 3, {2, 1, 3, 1});
        }

        // 测试非连续张量
        {
            auto tensor = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            auto transposed = tensor->permute({2, 0, 1});
            assert(!transposed->is_contiguous());

            auto unsqueezed = transposed->unsqueeze(2);
            assert(unsqueezed->get_shape().dims() == std::vector<index_t>({4, 2, 1, 3}));
            assert(unsqueezed->is_contiguous());
        }

        // 测试错误情况
        {
            const CPUTensor<float> tensor({2, 3, 4});

            ASSERT_THROWS({
                auto _ = tensor.unsqueeze(4);  // 维度超出范围
            }, std::out_of_range);

            ASSERT_THROWS({
              auto _ = tensor.unsqueeze(-5);  // 维度超出范围
            }, std::out_of_range);

        }

        std::cout << "All unsqueeze tests passed!" << std::endl;
    }


    static void test_squeeze() {
        // 辅助函数，用于检查 squeeze 操作的结果
        auto check_squeeze = [](const CPUTensor<float>& tensor,const std::optional<int> dim, const std::vector<index_t>& expected_shape) {
            const auto squeezed = dim ? tensor.squeeze(*dim) : tensor.squeeze();
            assert(squeezed->get_shape().dims() == expected_shape);
            std::cout << "Squeezed tensor shape (dim " << (dim ? std::to_string(*dim) : "all") << "): ";
            print_shape(squeezed->get_shape().dims());
            std::cout << std::endl;

            // 检查连续性
            assert(squeezed->is_contiguous());

            // 检查数据一致性
            assert(std::equal(tensor.data(), tensor.data() + tensor.num_elements(), squeezed->data()));
        };

        // 测试基本 squeeze
        {
            const CPUTensor<float> tensor({2, 1, 4, 1, 3});
            check_squeeze(tensor, 1, {2, 4, 1, 3});
            check_squeeze(tensor, 3, {2, 1, 4, 3});
            check_squeeze(tensor, std::nullopt, {2, 4, 3}); // squeeze all
        }

        // 测试 1D 张量
        {
            const CPUTensor<float> tensor({1});
            check_squeeze(tensor, 0, {});
            check_squeeze(tensor, std::nullopt, {});
        }

        // 测试 2D 张量
        {
            const CPUTensor<float> tensor({1, 3});
            check_squeeze(tensor, 0, {3});
            check_squeeze(tensor, 1, {1, 3});
            check_squeeze(tensor, std::nullopt, {3});
        }

        // 测试 3D 张量
        {
            const CPUTensor<float> tensor({2, 1, 3});
            check_squeeze(tensor, 1, {2, 3});
            check_squeeze(tensor, 0, {2, 1, 3});
            check_squeeze(tensor, std::nullopt, {2, 3});
        }

        // 测试 0D 张量 (标量)
        {
            const CPUTensor<float> tensor({});
            check_squeeze(tensor, std::nullopt, {});
        }

        // 测试非连续张量
        {
            auto tensor = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 1, 3, 1, 4});
            auto transposed = tensor->permute({4, 0, 2, 1, 3});
            assert(!transposed->is_contiguous());
            auto squeezed = transposed->squeeze();
            assert(squeezed->get_shape().dims() == std::vector<index_t>({4, 2, 3}));
            assert(!squeezed->is_contiguous());
            std::vector<std::vector<float>> expected = {
                { 0.,  4.,  8., 12., 16., 20. },
                { 1.,  5.,  9., 13., 17., 21.},
                {2.,  6., 10., 14., 18., 22. },
                { 3.,  7., 11., 15., 19., 23. }
            };
            auto x = squeezed->contiguous()->data();
            COMPARE_TENSOR_DATA(squeezed->contiguous()->data(), expected, 1e-6);
        }

        // 测试负索引
        {
            const CPUTensor<float> tensor({1, 2, 1, 3, 1});
            check_squeeze(tensor, -1, {1, 2, 1, 3});
            check_squeeze(tensor, -3, {1, 2, 3, 1});
            check_squeeze(tensor, -5, {2, 1, 3, 1});
        }

        // 测试多个连续的维度为1的情况
        {
            const CPUTensor<float> tensor({2, 1, 1, 1, 3});
            check_squeeze(tensor, std::nullopt, {2, 3});
        }

        // 测试所有维度都为1的情况
        {
            const CPUTensor<float> tensor({1, 1, 1});
            check_squeeze(tensor, std::nullopt, {});
        }

        // 测试错误情况
        {
            const CPUTensor<float> tensor({2, 3, 4});

            ASSERT_THROWS({
                auto _ = tensor.squeeze(3);  // 维度超出范围
            }, std::out_of_range);

            ASSERT_THROWS({
               auto _ = tensor.squeeze(-4);  // 维度超出范围
            }, std::out_of_range);

        }

        std::cout << "All squeeze tests passed!" << std::endl;
    }


    // 测试 view 操作
    static void test_view() {
        // 基本 view 操作
        CPUTensor<float> tensor1({2, 4, 3});
        const auto viewed_tensor1 = tensor1.view({4, 6});
        assert(viewed_tensor1->get_shape().dims() == std::vector<index_t>({4, 6}));

        // 使用 -1 自动计算维度
        CPUTensor<float> tensor2({3, 4, 5});
        const auto viewed_tensor2 = tensor2.view({6, -1});
        assert(viewed_tensor2->get_shape().dims() == std::vector<index_t>({6, 10}));

        // 增加维度
        CPUTensor<float> tensor3({12});
        const auto viewed_tensor3 = tensor3.view({3, 2, 2});
        assert(viewed_tensor3->get_shape().dims() == std::vector<index_t>({3, 2, 2}));


        // 减少维度
        CPUTensor<float> tensor4({2, 3, 4});
        const auto viewed_tensor4 = tensor4.view({24});
        assert(viewed_tensor4->get_shape().dims() == std::vector<index_t>({24}));


        // 保持相同的元素总数，但改变形状
        CPUTensor<float> tensor5({3, 8});
        const auto viewed_tensor5 = tensor5.view({4, 2, 3});
        assert(viewed_tensor5->get_shape().dims() == std::vector<index_t>({4, 2, 3}));


        ASSERT_THROWS({
            CPUTensor<float> tensor6({2, 3, 4});
            auto viewed_tensor6 = tensor6.view({5, 5});
        }, std::invalid_argument);


        // 测试非连续张量的 view
        auto tensor7 = CPUTensor<float>({4, 4},1.0);
        auto sliced_tensor7 = tensor7.slice({"0:1:3"});
        auto viewed_tensor7 = sliced_tensor7->view({4, 1});
        assert(viewed_tensor7->get_shape().dims() == std::vector<index_t>({4, 1}));


        ASSERT_THROWS({
           CPUTensor<float> tensor8({24});
            auto viewed_tensor8 = tensor8.view({-1, -1, 4});
         }, std::invalid_argument);


        // 测试 view 到空张量
        CPUTensor<float> tensor9({1, 1, 1});
        const auto viewed_tensor9 = tensor9.view({});
        assert(viewed_tensor9->get_shape().dims().empty());

        // 测试使用 0 作为维度（保持原始维度大小）
        CPUTensor<float> tensor10({2, 3, 4});
        const auto viewed_tensor10 = tensor10.view({2, 0, -1});
        assert(viewed_tensor10->get_shape().dims() == std::vector<index_t>({2, 3, 4}));

    }


    // 测试 reshape 操作
    static void test_reshape() {
        // 创建一个张量并填充数据
        CPUTensor<float> tensor1({2, 3, 4});
        tensor1.fill(1.0);

        // 重新定义张量的形状
        const auto reshaped_tensor1 = tensor1.reshape({4, 3, 2});
        assert(reshaped_tensor1->get_shape().dims() == std::vector<index_t>({4, 3, 2}));

        // 动态维度的重塑
        const auto reshaped_tensor2 = tensor1.reshape({-1, 6});
        assert(reshaped_tensor2->get_shape().dims() == std::vector<index_t>({4, 6}));

        // 另一种动态维度的重塑
        const auto reshaped_tensor3 = tensor1.reshape({6, -1});
        assert(reshaped_tensor3->get_shape().dims() == std::vector<index_t>({6, 4}));

        // 错误情况：不兼容的形状
        try {
            const auto reshaped_tensor4 = tensor1.reshape({3, 3, 3});
            assert(false && "Expected unsqueeze to throw an exception for out-of-bounds axis.");
        } catch (const std::invalid_argument& e) {
            std::cout << "Reshaped tensor1 (3, 3, 3):" << std::endl;
        }

        // 非连续张量的重塑
        const auto tensor2 = tensor1.slice({"0:1"});
        tensor2->fill(2.0);


        // 重新定义非连续张量的形状
        const auto reshaped_tensor5 = tensor2->reshape({2, -1});
        assert(reshaped_tensor5->get_shape().dims() == std::vector<index_t>({2, 6}));


        CPUTensor<float> tensor6({});
        tensor6.fill(6.0);
        // 另一种动态维度的重塑
        const auto reshaped_tensor6 = tensor6.reshape({1, 1, -1});
        assert(reshaped_tensor6->get_shape().dims() == std::vector<index_t>({1, 1, 1}));


    }

    // 测试非连续张量的操作
    static void test_non_contiguous() {
        CPUTensor<float> tensor2({2, 1});
        tensor2.fill(2.0);
        const auto tensor2_ex = tensor2.expand({2, 3});
        assert(tensor2_ex->get_shape().dims() == std::vector<index_t>({2, 3}));
        std::cout << "Expanded tensor2 shape: ";
        print_shape(tensor2.get_shape().dims());
        std::cout << std::endl;

        const auto unsqueezed_tensor2 = tensor2.unsqueeze(0);
        assert(unsqueezed_tensor2->get_shape().dims() == std::vector<index_t>({1, 2, 1}));
        std::cout << "Unsqueezed tensor2 shape: ";
        print_shape(unsqueezed_tensor2->get_shape().dims());
        std::cout << std::endl;

        const auto squeezed_tensor2 = unsqueezed_tensor2->squeeze(0);
        assert(squeezed_tensor2->get_shape().dims() == std::vector<index_t>({2, 1}));
        std::cout << "Squeezed tensor2 shape: ";
        print_shape(squeezed_tensor2->get_shape().dims());
        std::cout << std::endl;
    }

    static void test_cat() {
        // 基本测试
        {
            // 创建初始张量
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            // t1 现在是:
            // 0, 1, 2
            // 3, 4, 5

            // 创建 t2，与 t1 形状相同
            const auto t2 = CPUTensor<float>::arange(6, 12, 1.0)->view({2, 3});
            // t2 现在是:
            // 6, 7, 8
            // 9, 10, 11

            // 在第0维度上拼接
            const auto cat_dim0 = CPUTensor<float>::cat({t1.get(), t2.get()}, 0);
            assert(cat_dim0->get_shape().dims() == std::vector<index_t>({4, 3}));
            std::vector<float> expected_dim0 = {
                0.0f, 1.0f, 2.0f,
                3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f
            };
            for (index_t i = 0; i < expected_dim0.size(); ++i) {
                assert(std::abs(cat_dim0->mutable_data()[i] - expected_dim0[i]) < 1e-6f);
            }

            // 在第1维度上拼接
            const auto cat_dim1 = CPUTensor<float>::cat({t1.get(), t2.get()}, 1);
            assert(cat_dim1->get_shape().dims() == std::vector<index_t>({2, 6}));
            std::vector<float> expected_dim1 = {
                0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
                3.0f, 4.0f, 5.0f, 9.0f, 10.0f, 11.0f
            };
            for (index_t i = 0; i < expected_dim1.size(); ++i) {
                assert(std::abs(cat_dim1->mutable_data()[i] - expected_dim1[i]) < 1e-6f);
            }

            // 测试非连续张量
            const auto t3 = t1->permute({1, 0})->contiguous();  // t3 是 3x2
            const auto t4 = t2->permute({1, 0})->contiguous();  // t4 也是 3x2

            // 在第1维度上拼接非连续张量
            const auto cat_non_contiguous = CPUTensor<float>::cat({t3.get(), t4.get()}, 1);
            assert(cat_non_contiguous->get_shape().dims() == std::vector<index_t>({3, 4}));
            std::vector<float> expected_non_contiguous = {
                0.0f, 3.0f, 6.0f, 9.0f,
                1.0f, 4.0f, 7.0f, 10.0f,
                2.0f, 5.0f, 8.0f, 11.0f
            };
            for (index_t i = 0; i < expected_non_contiguous.size(); ++i) {
                assert(std::abs(cat_non_contiguous->mutable_data()[i] - expected_non_contiguous[i]) < 1e-6f);
            }
        }

        // 1D张量拼接测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 3, 1.0);
            const auto t2 = CPUTensor<float>::arange(3, 6, 1.0);

            const auto cat_1d = CPUTensor<float>::cat({t1.get(), t2.get()}, 0);
            assert(cat_1d->get_shape().dims() == std::vector<index_t>({6}));
            for (int i = 0; i < 6; ++i) {
                assert(cat_1d->mutable_data()[i] == static_cast<float>(i));
            }
        }

        // 3D张量拼接测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = CPUTensor<float>::arange(24, 48, 1.0)->view({2, 3, 4});

            // 在第0维度上拼接
            const auto cat_3d_dim0 = CPUTensor<float>::cat({t1.get(), t2.get()}, 0);
            assert(cat_3d_dim0->get_shape().dims() == std::vector<index_t>({4, 3, 4}));

            // 在第1维度上拼接
            const auto cat_3d_dim1 = CPUTensor<float>::cat({t1.get(), t2.get()}, 1);
            assert(cat_3d_dim1->get_shape().dims() == std::vector<index_t>({2, 6, 4}));

            // 在第2维度上拼接
            const auto cat_3d_dim2 = CPUTensor<float>::cat({t1.get(), t2.get()}, 2);
            assert(cat_3d_dim2->get_shape().dims() == std::vector<index_t>({2, 3, 8}));
        }
        // 2D张量拼接测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(0, 4, 1.0)->view({2, 2});  // 形状不匹配
            const auto cat_2d_dim1 = CPUTensor<float>::cat({t1.get(), t2.get()}, 1);
            assert(cat_2d_dim1->get_shape().dims() == std::vector<index_t>({2, 5}));
            std::cout << *cat_2d_dim1 << std::endl;
            assert(cat_2d_dim1->mutable_data()[0] == 0.0f && cat_2d_dim1->mutable_data()[1] == 1.0f && cat_2d_dim1->mutable_data()[2] == 2.0f
                && cat_2d_dim1->mutable_data()[3] == 0.0f && cat_2d_dim1->mutable_data()[4] == 1.0f);

            assert(cat_2d_dim1->mutable_data()[5] == 3.0f && cat_2d_dim1->mutable_data()[6] == 4.0f && cat_2d_dim1->mutable_data()[7] == 5.0f
                       && cat_2d_dim1->mutable_data()[8] == 2.0f && cat_2d_dim1->mutable_data()[9] == 3.0f);
        }

        //错误处理测试
        ASSERT_THROWS({
          const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
          const auto t2 = CPUTensor<float>::arange(0, 6, 1.0)->view({3, 2});  // 完全不同的形状
          CPUTensor<float>::cat({t1.get(), t2.get()}, 1);  // 这里应该抛出异常
      }, std::invalid_argument);

        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(6, 12, 1.0)->view({2, 3});

            CPUTensor<float>::cat({t1.get(), t2.get()}, 3);  // 无效的维度
        }, std::invalid_argument);

        ASSERT_THROWS({
            std::vector<Tensor<float>*> tensors;  // 空向量
            CPUTensor<float>::cat(tensors, 0);
        }, std::invalid_argument);

        // 标量拼接测试（应报错）
        ASSERT_THROWS({
            CPUTensor<float> scalar1({});
            scalar1.fill(2.0);
            CPUTensor<float> scalar2({});
            scalar2.fill(1.0);
            CPUTensor<float>::cat({&scalar1, &scalar2}, 0);
        }, std::invalid_argument);


        {
            // n维度非连续拼接
            CPUTensor<float> tensor3d_1({2, 3, 4}, 1.0);
            CPUTensor<float> tensor3d_2({2, 3, 4}, 2.0);

            // Slice to create non-contiguous tensors
            const auto sliced_tensor1 = tensor3d_1.slice({"0:2", "1:3"});
            const auto sliced_tensor2 = tensor3d_2.slice({"0:2", "1:3"});
            // Concatenate along a specific axis
            const auto cat_non_contiguous = CPUTensor<float>::cat({sliced_tensor1.get(), sliced_tensor2.get()}, 1);
            // Check the shape
            assert(cat_non_contiguous->get_shape().dims() == std::vector<index_t>({2, 4, 4}));
            std::vector<std::vector<float>> expected = {
                {1., 1., 1., 1.},
                {1., 1., 1., 1.},
                {2., 2., 2., 2.},
                {2., 2., 2., 2.},
                {1., 1., 1., 1.},
                {1., 1., 1., 1.},
                {2., 2., 2., 2.},
                {2., 2., 2., 2.}
            };
            COMPARE_TENSOR_DATA(cat_non_contiguous->mutable_data(),expected, 1e-6);
        }

        {
            // Create a 1D tensor
            CPUTensor<float> tensor1d({6});
            for (int i = 0; i < 6; ++i) {
                tensor1d.set_value({static_cast<index_t>(i)}, static_cast<float>(i));
            }
            // Slice to create non-contiguous tensors (e.g., take every other element)
            const auto sliced_tensor3 = tensor1d.slice({"5:-1:-2"}); // [5, 3, 1]
            const auto sliced_tensor4 = tensor1d.slice({"4:-1:-2"}); // [6, 4, 2]

            // Concatenate the slices
            const auto cat_non_contiguous_1d = CPUTensor<float>::cat({sliced_tensor3.get(), sliced_tensor4.get()}, 0);

            // Check the shape
            assert(cat_non_contiguous_1d->get_shape().dims() == std::vector<index_t>({6}));

            std::vector<std::vector<float>> expected = {
                {5., 3., 1., 4. ,2. ,0.},
            };
            std::cout << *cat_non_contiguous_1d << std::endl;
            COMPARE_TENSOR_DATA(cat_non_contiguous_1d->mutable_data(),expected, 1e-6);
        }
        std::cout << "All cat tests passed successfully!" << std::endl;
    }


    static void test_clone() {
        // 1维度拼接
        const auto tensor1 = CPUTensor<float>::arange(0,20,1);
        std::cout << *tensor1 << std::endl;
        const auto tensor1_slice = tensor1->slice({"20:10:-2"});
        std::cout << *tensor1_slice << std::endl;
        const auto tensor1_slice_clone = tensor1_slice->clone();
        std::cout << *tensor1_slice_clone << std::endl;
        std::cout << "1D non-contiguous tensor clone result: ";
        assert(tensor1_slice_clone->get_shape().dims() == std::vector<index_t>({5}));
        std::cout << std::endl;

        // 1维度拼接
        auto tensor2 = CPUTensor<float>({20},1.5);
        std::cout << tensor2 << std::endl;
        const auto tensor2_slice = tensor2.slice({"20:10:-2"});
        std::cout << *tensor2_slice << std::endl;
        const auto tensor2_slice_clone = tensor2_slice->clone();
        std::cout << *tensor2_slice_clone << std::endl;
        std::cout << "1D non-contiguous tensor clone result: ";
        assert(tensor2_slice_clone->get_shape().dims() == std::vector<index_t>({5}));
        std::cout << std::endl;

    }

    static void test_transpose() {
        // Transpose a 2D tensor
        auto tensor2d = CPUTensor<float>::arange(0,12,1.0)->view({3,4});
        const auto transposed_2d = tensor2d->transpose(0, 1);
        std::cout << *transposed_2d << std::endl;
        assert(transposed_2d->get_shape().dims() == std::vector<index_t>({4, 3}));
        const auto transposed_2d_c = transposed_2d->clone();
        std::cout << *transposed_2d_c << std::endl;
        std::cout << "2D transpose result: ";
        print_shape(transposed_2d->get_shape().dims());
        std::cout << std::endl;
        std::cout << *tensor2d << std::endl;

        // Transpose a 3D tensor
        const CPUTensor<float> tensor3d({2, 3, 4},2.0);
        const auto transposed_3d = tensor3d.transpose(1, 2);
        assert(transposed_3d->get_shape().dims() == std::vector<index_t>({2, 4, 3}));
        std::cout << "3D transpose result: ";
        print_shape(transposed_3d->get_shape().dims());
        std::cout << std::endl;

        // Transpose a 1D tensor (should be a no-op)
        CPUTensor<float> tensor1d({5});
        tensor1d.fill(3.0);
        const auto transposed_1d = tensor1d.transpose(0, 0);
        assert(transposed_1d->get_shape().dims() == std::vector<index_t>({5}));
        std::cout << "1D transpose result (no-op): ";
        print_shape(transposed_1d->get_shape().dims());
        std::cout << std::endl;

        // Transpose a 4D tensor
        CPUTensor<float> tensor4d({2, 3, 4, 5});
        tensor4d.fill(4.0);
        const auto transposed_4d = tensor4d.transpose(0, 3);
        assert(transposed_4d->get_shape().dims() == std::vector<index_t>({5, 3, 4, 2}));
        std::cout << "4D transpose result: ";
        print_shape(transposed_4d->get_shape().dims());
        std::cout << std::endl;

        // Error case: Invalid axes
        try {
            auto _ = tensor2d->transpose(0, 2);
            assert(false && "Expected transpose to throw an exception for invalid axes.");
        } catch (const std::out_of_range& e) {
            std::cout << "Caught expected error for invalid transpose axes: " << e.what() << std::endl;
        }
    }

    static void test_omp() {
        // MEASURE_TIME(auto  a1 = CPUTensor<float>::arrange(1,100000000,1.0););
        const auto a1 = CPUTensor<float>::arange(1,10000000,1.0)->expand({250,-1})->slice({":","::2"});
        // MEASURE_TIME(auto x = a1->clone());
        const auto x = a1->clone();
        const auto data_bytes = x->n_bytes();
        const double megabytes = static_cast<double>(data_bytes) / (1024.0 * 1024.0);
        std::cout << Utils::Format("Tensor memory size on CPU (megabytes): {%d} mb",megabytes)  << std::endl;
    }

    static void test_permute() {
      {

          const auto t1 = CPUTensor<float>::arange(0,24,1.0)->view({2,3,4});
          const auto t2 = t1 ->permute({-1,-2,-3});
          std::vector<std::vector<float>> expected = {
              { 0., 12., 4., 16., 8., 20.},
              { 1., 13., 5., 17., 9., 21.},
              { 2., 14., 6., 18., 10., 22.},
              { 3., 15., 7., 19., 11., 23.},
          };
          const auto t3 = t2->contiguous();
          assert(t3->get_shape().dims() == std::vector<index_t>({4, 3, 2}));
          assert(t3 != t2);
          assert(t3->is_contiguous());
          COMPARE_TENSOR_DATA(t3->mutable_data(), expected, 0);

      }

            // 基本的维度重排测试
        {
          const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
          const auto t2 = t1->permute({2, 0, 1});
          assert(t2->get_shape().dims() == std::vector<index_t>({4, 2, 3}));
          std::vector<std::vector<float>> expected = {
              { 0.,  4.,  8., 12., 16., 20.},
              { 1.,  5.,  9., 13., 17., 21.},
              { 2.,  6., 10., 14., 18., 22.},
              { 3.,  7., 11., 15., 19., 23.},
          };
          COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        // 负数维度测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({-1, -3, -2});
            std::vector<std::vector<float>> expected = {
              { 0.,  4.,  8., 12., 16., 20.},
              { 1.,  5.,  9., 13., 17., 21.},
              { 2.,  6., 10., 14., 18., 22.},
              { 3.,  7., 11., 15., 19., 23.},
          };
          COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        // 恒等置换测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({0, 1, 2});
            assert(t2->get_shape().dims() == t1->get_shape().dims());
            assert(t2->data() == t1->data());
            assert(t2->get_strides() == t1->get_strides());
        }

        // 一维张量测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 5, 1.0);
            const auto t2 = t1->permute({0});
            assert(t2->get_shape().dims() == t1->get_shape().dims());
            assert(t2->data() == t1->data());
            assert(t2->get_strides() == t1->get_strides());
        }

        // 高维张量测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 120, 1.0)->view({2, 3, 4, 5});
          const auto t2 = t1->permute({3, 1, 2, 0});
          assert(t2->get_shape().dims() == std::vector<index_t>({5, 3, 4, 2}));
          std::vector<std::vector<float>> expected ={
              { 0.,  60., 5.,  65., 10.,  70., 15.,  75., 20.,  80., 25.,  85., 30., 90., 35.,  95., 40., 100., 45., 105., 50., 110., 55., 115.},
              { 1.,  61., 6.,  66., 11.,  71., 16.,  76., 21.,  81., 26.,  86., 31.,  91., 36.,  96., 41., 101., 46., 106., 51., 111., 56., 116.},
              { 2.,  62., 7.,  67., 12.,  72., 17.,  77., 22.,  82., 27.,  87., 32.,  92., 37.,  97., 42., 102., 47., 107., 52., 112., 57., 117.},
              { 3.,  63., 8.,  68., 13.,  73., 18.,  78., 23.,  83., 28.,  88., 33.,  93., 38.,  98., 43., 103., 48., 108., 53., 113., 58., 118.},
              { 4.,  64., 9.,  69., 14.,  74., 19.,  79., 24.,  84., 29.,  89., 34.,  94., 39.,  99., 44., 104., 49., 109., 54., 114., 59., 119.}
          };

          COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        // 维度不匹配错误测试
        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({0, 1});
        }, std::invalid_argument);

        // 无效维度错误测试
        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({0, 1, 3});
        }, std::out_of_range);

        // 重复维度错误测试
        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({0, 1, 1});
        }, std::invalid_argument);

        // 大的负数维度错误测试
        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({0, 1, -4});
        }, std::out_of_range);

        // 数据共享测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({2, 0, 1});
            t1->mutable_data()[0] = 100.0f;
            assert(t2->mutable_data()[0] == 100.0f);
        }

        // contiguous 测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = t1->permute({2, 1, 0});
            const auto t3 = t2->contiguous();
            assert(t3->get_shape().dims() == std::vector<index_t>({4, 3, 2}));
            assert(t3 != t2);
            assert(t3->is_contiguous());
        }

        // 一维张量的 -1 维度测试
      {
          const auto t1 = CPUTensor<float>::arange(0, 1, 1.0);
          const auto t2 = t1->permute({-1});
          // 这里不应该抛出异常，因为对一维张量使用 -1 是有效的
          assert(t1->get_shape().dims() == t2->get_shape().dims());
          assert(t2->is_contiguous());
          assert(t2->data() == t1->data());

      }

      {
          // 创建一个 2x3x1 的张量
          auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3, 1});
          std::cout << "Original tensor:" << *t1 << std::endl;

          // 扩展到 2x3x4
          auto t2 = t1->expand({2, 3, 4});
          std::cout << "\nExpanded tensor:" << *t2  << std::endl;

          // 验证扩展后的形状
          assert(t2->get_shape().dims() == std::vector<index_t>({2, 3, 4}));

          // 验证扩展后的数据
          for (index_t i = 0; i < 2; ++i) {
              for (index_t j = 0; j < 3; ++j) {
                  float expected_value = i * 3 + j;
                  for (index_t k = 0; k < 4; ++k) {
                      assert(t2->at({i,j,k}) == expected_value);
                  }
              }
          }

          // 执行 permute 操作，交换维度 -1, -2, -3 (等价于 2, 1, 0)
          auto t3 = t2->permute({-1, -2, -3})->contiguous();
          std::cout << "\nPermuted tensor:" << *t3 << std::endl;

          // 验证 permute 后的形状
          assert(t3->get_shape().dims() == std::vector<index_t>({4, 3, 2}));

          // 验证 permute 后的数据
          for (index_t i = 0; i < 4; ++i) {
              for (index_t j = 0; j < 3; ++j) {
                  for (index_t k = 0; k < 2; ++k) {
                      float expected_value = k * 3 + j;
                      assert(t3->at({i,j,k}) == expected_value);
                  }
              }
          }

      }

        std::cout << "All permute tests passed!" << std::endl;

    }


    static void test_flatten() {
        // 测试0维张量
        {
            const auto t0d = CPUTensor<float>::scalar(5.0);
            const auto flattened = t0d->flatten();
            assert(flattened->get_shape().dims() == std::vector<index_t>({}));
            assert(flattened->mutable_data()[0] == 5.0f);
        }

        // 测试1维张量
        {
            const auto t1d = CPUTensor<float>::arange(0, 5, 1.0);
            const auto flattened = t1d->flatten();
            assert(flattened->get_shape().dims() == std::vector<index_t>({5}));
            assert(flattened->mutable_data() == t1d->mutable_data());
            std::vector<std::vector<int>> expected = {
                {0, 1, 2, 3, 4},
            };
            COMPARE_TENSOR_DATA(t1d->mutable_data(), expected, 0);
        }

        // 测试2维张量
        {
            const auto t2d = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto flattened = t2d->flatten();
            assert(flattened->get_shape().dims() == std::vector<index_t>({6}));
            assert(flattened->mutable_data() == t2d->mutable_data());
            std::vector<std::vector<int>> expected = {
                {0, 1, 2, 3, 4, 5},
            };
            COMPARE_TENSOR_DATA(flattened->mutable_data(), expected, 0);
        }

        // 测试3维张量
        {
            const auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto flattened = t3d->flatten();
            assert(flattened->get_shape().dims() == std::vector<index_t>({24}));
            std::vector<std::vector<float>> expected = {{}};
            for (index_t i =0; i < 24; ++i) {
                expected[0].push_back(static_cast<float>(i));
            }
            assert(flattened->mutable_data() == t3d->mutable_data());
            COMPARE_TENSOR_DATA(flattened->mutable_data(), expected, 0);
        }

        // 测试非连续张量
        {
            const auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t3d_permuted = t3d->permute({2, 1, 0});
            const auto flattened = t3d_permuted->flatten();
            assert(flattened->get_shape().dims() == std::vector<index_t>({24}));
            assert(flattened->mutable_data() != t3d_permuted->mutable_data());
            std::cout << *t3d_permuted << std::endl;
            assert(flattened->is_contiguous());
        }

        // 测试指定start_dim和end_dim
        {
            const auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});

            // flatten(0, 1)
            const auto flattened01 = t3d->flatten(0, 1);
            assert(flattened01->get_shape().dims() == std::vector<index_t>({6, 4}));
            assert(flattened01->mutable_data() == t3d->mutable_data());
            std::vector<std::vector<float>> expected1 = {
                {0.,  1.,  2.,  3.},
                {4.,  5.,  6.,  7.},
                {8.,  9., 10., 11.},
                {12., 13., 14., 15.},
                {16., 17., 18., 19.},
                {20., 21., 22., 23.},
            };
            COMPARE_TENSOR_DATA(flattened01->mutable_data(), expected1, 0);

            // flatten(1, 2)
            const auto flattened12 = t3d->flatten(1, 2);
            assert(flattened12->get_shape().dims() == std::vector<index_t>({2, 12}));
            assert(flattened12->mutable_data() == t3d->mutable_data());

            std::vector<std::vector<float>> expected2 = {
                {0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.},
                {12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.},
            };
            COMPARE_TENSOR_DATA(flattened12->mutable_data(), expected2, 0);
        }

        // 测试使用负数索引
        {
            const auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});

            // flatten(-2, -1)
            const auto flattened = t3d->flatten(-2, -1);
            assert(flattened->get_shape().dims() == std::vector<index_t>({2, 12}));
            assert(flattened->mutable_data() == t3d->mutable_data());
            std::vector<std::vector<float>> expected2 = {
                {0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.},
                {12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.},
            };
            COMPARE_TENSOR_DATA(flattened->mutable_data(), expected2, 0);
        }

        // 测试边界情况
        {
            const auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});

            // flatten(0, -1) 等同于完全flatten
            const auto flattened = t3d->flatten(0, -1);
            assert(flattened->get_shape().dims() == std::vector<index_t>({24}));
            assert(flattened->mutable_data() == t3d->mutable_data());
            std::vector<std::vector<float>> expected = {
                { 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.},
            };
            COMPARE_TENSOR_DATA(flattened->mutable_data(), expected, 0);
        }

        // 测试非法输入
        {

            auto t3d = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            ASSERT_THROWS({
                auto _ = t3d->flatten(2, 1);
            }, std::out_of_range);

            ASSERT_THROWS({
                auto _ = t3d->flatten(0, 3);
            }, std::out_of_range);

            ASSERT_THROWS({
                auto _ =t3d->flatten(-4, -1);
            }, std::out_of_range);

        }


        {
            // 创建一个 1x2x8 的张量，填充值为 2.0
            const auto tensor = CPUTensor<float>({1, 2, 8}, 2.0);
            // 执行 expand 操作
            const auto expanded = tensor.expand({16, 2, 8});
            // 模拟 expanded[0::2,:,:]
            const auto tensor1 = expanded->slice({"::2"});
            // 第一种情况：flatten(0,1)，预期会产生复制
            const auto x = tensor1->flatten(0, 1);
            // 断言
            assert(x->get_shape().dims() == std::vector<index_t>({16, 8}));
            assert(x->is_contiguous());
            assert(x->get_strides() == std::vector<index_t>({8, 1}));
            assert(x->mutable_data() != tensor1->mutable_data());  // 数据应该被复制

            // 第二种情况：flatten(0,1)，预期不会产生复制
            const auto x1 = tensor1->flatten(1, 2);
            // 断言
            assert(x1->get_shape().dims() == std::vector<index_t>({8, 16}));
            assert(!x1->is_contiguous());
            assert(x1->get_strides() == std::vector<index_t>({0, 1}));
            assert(x1->mutable_data() == tensor1->mutable_data());  // 数据不应该被复制

        }
        std::cout << "All flatten tests passed!" << std::endl;
    }

    static void test_randn() {
        // const auto random_tensor1 = CPUTensor<float>::randn({2, 3, 4});
        // std::cout << *random_tensor1 << std::endl;
        // const auto random_tensor2 = CPUTensor<float>::randn({2, 4, 7});
        //
        // auto random_tensor3 =  random_tensor1->matmul(*random_tensor2);
        // auto x = CPUTensor<float>::isSafeToModify(random_tensor3);
        // std::cout << *random_tensor3 << std::endl;

    }

    static void test_stack(){
        // 基本 stack 操作测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(6, 12, 1.0)->view({2, 3});
            const auto t3 = CPUTensor<float>::arange(12, 18, 1.0)->view({2, 3});

            // 在第 0 维 stack
            const auto stacked0 = CPUTensor<float>::stack({t1.get(), t2.get(), t3.get()}, 0);
            assert(stacked0->get_shape().dims() == std::vector<index_t>({3, 2, 3}));
            assert(stacked0->mutable_data()[0] == 0.0f && stacked0->mutable_data()[6] == 6.0f && stacked0->mutable_data()[12] == 12.0f);

            // 在第 1 维 stack
            const auto stacked1 = CPUTensor<float>::stack({t1.get(), t2.get(), t3.get()}, 1);
            std::cout << *stacked1 << std::endl;
            assert(stacked1->get_shape().dims() == std::vector<index_t>({2, 3, 3}));
            assert(stacked1->mutable_data()[0] == 0.0f && stacked1->mutable_data()[3] == 6.0f && stacked1->mutable_data()[6] == 12.0f);

            // 在第 2 维 stack
            const auto stacked2 = CPUTensor<float>::stack({t1.get(), t2.get(), t3.get()}, 2);
            assert(stacked2->get_shape().dims() == std::vector<index_t>({2, 3, 3}));
            std::cout << *stacked2 << std::endl;
            assert(stacked2->mutable_data()[0] == 0.0f && stacked2->mutable_data()[1] == 6.0f && stacked2->mutable_data()[2] == 12.0f);
        }

        // 高维张量 stack 测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 24, 1.0)->view({2, 3, 4});
            const auto t2 = CPUTensor<float>::arange(24, 48, 1.0)->view({2, 3, 4});

            const auto stacked = CPUTensor<float>::stack({t1.get(), t2.get()}, 1);
            assert(stacked->get_shape().dims() == std::vector<index_t>({2, 2, 3, 4}));
            assert(stacked->mutable_data()[0] == 0.0f && stacked->mutable_data()[24] == 12.0f);
        }

        // 负数维度测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(6, 12, 1.0)->view({2, 3});


            const auto stacked = CPUTensor<float>::stack({t1.get(), t2.get()}, -1);  // 应该等同于 dim=2
            assert(stacked->get_shape().dims() == std::vector<index_t>({2, 3, 2}));
        }

        // 非连续张量测试
        {
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = t1->permute({1, 0})->contiguous()->view({2, 3});  // 非连续张量，但形状相同

            const auto stacked = CPUTensor<float>::stack({t1.get(), t2.get()}, 0);
            assert(stacked->get_shape().dims() == std::vector<index_t>({2, 2, 3}));
            assert(stacked->mutable_data()[0] == 0.0f && stacked->mutable_data()[6] == 0.0f);
            assert(stacked->mutable_data()[1] == 1.0f && stacked->mutable_data()[7] == 3.0f);
            assert(stacked->mutable_data()[2] == 2.0f && stacked->mutable_data()[8] == 1.0f);
            assert(stacked->mutable_data()[3] == 3.0f && stacked->mutable_data()[9] == 4.0f);
            assert(stacked->mutable_data()[4] == 4.0f && stacked->mutable_data()[10] == 2.0f);
            assert(stacked->mutable_data()[5] == 5.0f && stacked->mutable_data()[11] == 5.0f);
        }

        // 错误处理测试
        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(0, 4, 1.0)->view({2, 2});  // 形状不匹配

            CPUTensor<float>::stack({t1.get(), t2.get()}, 0);
        }, std::invalid_argument);

        ASSERT_THROWS({
            const auto t1 = CPUTensor<float>::arange(0, 6, 1.0)->view({2, 3});
            const auto t2 = CPUTensor<float>::arange(6, 12, 1.0)->view({2, 3});


            CPUTensor<float>::stack({t1.get(), t2.get()}, 3);  // 无效的维度
        }, std::invalid_argument);

        ASSERT_THROWS({
            const std::vector<Tensor<float>*> tensors;  // 空向量
            CPUTensor<float>::stack(tensors, 0);
        }, std::invalid_argument);

        {
            const auto t1 = CPUTensor<float>::arange(0,128,1)->view({2,8,8})->slice({":",":","::2"});
            const auto t2 = CPUTensor<float>::arange(128,256,1)->view({2,8,8})->slice({":",":","::2"});
            const auto t3 = CPUTensor<float>::arange(256,384,1)->view({2,8,8})->slice({":",":","::2"});
            const auto stacked0 = CPUTensor<float>::stack({t1.get(),t2.get(),t3.get()}, 1);
            assert(stacked0->get_shape().dims() == std::vector<index_t>({2, 3, 8, 4  }));
            assert(stacked0->mutable_data()[0] == 0.0f && stacked0->mutable_data()[1] == 2.0f && stacked0->mutable_data()[2] == 4.0f && stacked0->mutable_data()[3] == 6.0f);
            assert(stacked0->mutable_data()[28] == 56.0f && stacked0->mutable_data()[29] == 58.0f && stacked0->mutable_data()[30] == 60.0f && stacked0->mutable_data()[31] == 62.0f);

            assert(stacked0->mutable_data()[32] == 128.0f && stacked0->mutable_data()[33] == 130.0f && stacked0->mutable_data()[34] == 132.0f && stacked0->mutable_data()[35] == 134.0f);
            assert(stacked0->mutable_data()[60] == 184.0f && stacked0->mutable_data()[61] == 186.0f && stacked0->mutable_data()[62] == 188.0f && stacked0->mutable_data()[63] == 190.0f);

            assert(stacked0->mutable_data()[160] == 320.0f && stacked0->mutable_data()[161] == 322.0f && stacked0->mutable_data()[162] == 324.0f && stacked0->mutable_data()[163] == 326.0f);
            assert(stacked0->mutable_data()[188] == 376.0f && stacked0->mutable_data()[189] == 378.0f && stacked0->mutable_data()[190] == 380.0f && stacked0->mutable_data()[191] == 382.0f);
        }

        {
            // 创建随机数生成器
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);

            // 定义 5 维张量的形状
            std::vector<index_t> shape = {2, 3, 4, 5, 6};

            // 创建 3 个输入张量
            std::vector<std::shared_ptr<CPUTensor<float>>> input_tensors;
            for (int i = 0; i < 3; ++i) {
                auto tensor = std::make_shared<CPUTensor<float>>(shape);
                // 用随机数填充张量
                for (index_t j = 0; j < tensor->size(); ++j) {
                    tensor->mutable_data()[j] = static_cast<float>(dis(gen));
                }
                input_tensors.push_back(tensor);
            }

            // 执行 stack 操作
            const auto result = CPUTensor<float>::stack({input_tensors[0].get(),
                input_tensors[1].get(), input_tensors[2].get()}, 1);

            // 验证结果形状
            const std::vector<index_t> expected_shape = {2, 3, 3, 4, 5, 6};  // 在第 2 维上 stack 后的预期形状
            assert(result->get_shape().dims() == expected_shape);

            // 验证结果内容
            for (index_t i = 0; i < shape[0]; ++i) {
                for (index_t j = 0; j < 3; ++j) {  // 3 是输入张量的数量
                    for (index_t k = 0; k < shape[1]; ++k) {
                        for (index_t l = 0; l < shape[2]; ++l) {
                            for (index_t m = 0; m < shape[3]; ++m) {
                                for (index_t n = 0; n < shape[4]; ++n) {
                                    const index_t input_index = ((((i * shape[1] + k) * shape[2] + l) * shape[3] + m) * shape[4] + n);
                                    const index_t result_index = ((((i * 9 + (j * shape[1] + k)) * shape[2] + l) * shape[3] + m) * shape[4] + n);
                                    assert(std::abs(result->mutable_data()[result_index] - input_tensors[j]->mutable_data()[input_index]) < 1e-6);
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    static void test_repeat() {

        // // 1维张量测试
        {
            auto t1 = CPUTensor<float>::arange(0,3,1);
            const auto result = t1->repeat({2});
            ASSERT_EQ_VECTOR(result->get_shape().dims(), std::vector<index_t>({6}));
            std::vector<std::vector<int>> expected = {
                {0, 1, 2, 0, 1, 2}
            };
            COMPARE_TENSOR_DATA(result->mutable_data(), expected, 0);
        }

        // 2维张量测试
        {
            CPUTensor<float> tensor2d({2, 2});
            tensor2d.set_value({0, 0}, 1);
            tensor2d.set_value({0, 1}, 2);
            tensor2d.set_value({1, 0}, 3);
            tensor2d.set_value({1, 1}, 4);
            auto result = tensor2d.repeat({2, 3});
            ASSERT_EQ_VECTOR(result->get_shape().dims(), std::vector<index_t>({4, 6}));
            std::vector<std::vector<int>> expected = {
                {1, 2, 1, 2, 1, 2},
                {3, 4, 3, 4, 3, 4},
                {1, 2, 1, 2, 1, 2},
                {3, 4, 3, 4, 3, 4}
            };
            COMPARE_TENSOR_DATA(result->mutable_data(), expected, 0);
        }


        // 标量无法复制
        ASSERT_THROWS({
            auto t1 = CPUTensor<float>::scalar(1);
            auto _ = t1->repeat({2}); // 维度数量不匹配
        }, std::invalid_argument);

        // 维度不匹配测试
        ASSERT_THROWS({
            CPUTensor<float> tensor({2, 3});
            auto _ = tensor.repeat({2}); // 维度数量不匹配
        }, std::invalid_argument);

        // 零重复次数测试
        ASSERT_THROWS({
            CPUTensor<float> tensor({2, 3});
            auto _ = tensor.repeat({2, 0});
        }, std::invalid_argument);

        // Case 3: 2D tensor, repeat with more dimensions
        {
            auto t1 = CPUTensor<float>::arange(0,12,1)->view({2,3,2});
            auto result = t1->repeat({2, 2, 2, 2, 2});
            std::vector<std::vector<float>> expected =  {
                { 0,  1,  0,  1},{ 2,  3,  2,  3},{ 4,  5,  4,  5},{ 0,  1,  0,  1},
                { 2,  3,  2,  3},{ 4,  5,  4,  5},{ 6,  7,  6,  7},{ 8,  9,  8,  9},
                { 10, 11, 10, 11},{ 6,  7,  6,  7},{ 8,  9,  8,  9},{ 10, 11, 10, 11},
                { 0,  1,  0,  1},{ 2,  3,  2,  3},{ 4,  5,  4,  5},{ 0,  1,  0,  1},
                { 2,  3,  2,  3},{ 4,  5,  4,  5},{ 6,  7,  6,  7},{ 8,  9,  8,  9},
                { 10, 11, 10, 11},{ 6,  7,  6,  7},{ 8,  9,  8,  9},{ 10, 11, 10, 11},
                { 0,  1,  0,  1},{ 2,  3,  2,  3},{ 4,  5,  4,  5},{ 0,  1,  0,  1},
                { 2,  3,  2,  3},{ 4,  5,  4,  5},{ 6,  7,  6,  7},{ 8,  9,  8,  9},
                { 10, 11, 10, 11},{ 6,  7,  6,  7},{ 8,  9,  8,  9},{ 10, 11, 10, 11},
                { 0,  1,  0,  1},{ 2,  3,  2,  3},{ 4,  5,  4,  5},{ 0,  1,  0,  1},
                { 2,  3,  2,  3},{ 4,  5,  4,  5},{ 6,  7,  6,  7},{ 8,  9,  8,  9},
                {10, 11, 10, 11},{ 6,  7,  6,  7},{ 8,  9,  8,  9},{ 10, 11, 10, 11}

            };
            COMPARE_TENSOR_DATA(result->mutable_data(), expected, 1e-6);
        }

        // 非连续张量的repeat测试
        {
            CPUTensor<float> tensor3d({3, 4, 5});
            for (int i = 0; i < 3 * 4 * 5; ++i) {
                tensor3d.mutable_data()[i] = static_cast<float>(i);
            }
            auto sliced_tensor = tensor3d.slice({"1:3", "0:4:2", "1:4"});
            auto result = sliced_tensor->repeat({2, 2, 1});
            ASSERT_EQ_VECTOR(result->get_shape().dims(), std::vector<index_t>({4, 4, 3}));
            std::vector<std::vector<float>> expected = {
                {21, 22, 23}, {31, 32, 33}, {21, 22, 23}, {31, 32, 33},
                {41, 42, 43}, {51, 52, 53}, {41, 42, 43}, {51, 52, 53},
                {21, 22, 23}, {31, 32, 33}, {21, 22, 23}, {31, 32, 33},
                {41, 42, 43}, {51, 52, 53}, {41, 42, 43}, {51, 52, 53}
            };
            COMPARE_TENSOR_DATA(result->mutable_data(), expected, 1e-6);
        }

        //非连续 负数索引 repeat 测试

        {

            auto t1 = CPUTensor<float>::arange(0,96,1)->view({4, 6, 4})->slice({"::-2",{"::-2"},{"::-2"}});
            auto result =  t1->repeat({2,2,3,2});
            std::vector<std::vector<float>> expected = {
                {95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},{95, 93, 95, 93},
                {87, 85, 87, 85},{79, 77, 79, 77},{95, 93, 95, 93},{87, 85, 87, 85},
                {79, 77, 79, 77},{47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},
                {47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},{47, 45, 47, 45},
                {39, 37, 39, 37},{31, 29, 31, 29},{95, 93, 95, 93},{87, 85, 87, 85},
                {79, 77, 79, 77},{95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},
                {95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},{47, 45, 47, 45},
                {39, 37, 39, 37},{31, 29, 31, 29},{47, 45, 47, 45},{39, 37, 39, 37},
                {31, 29, 31, 29},{47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},
                {95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},{95, 93, 95, 93},
                {87, 85, 87, 85},{79, 77, 79, 77},{95, 93, 95, 93},{87, 85, 87, 85},
                {79, 77, 79, 77},{47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},
                {47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},{47, 45, 47, 45},
                {39, 37, 39, 37},{31, 29, 31, 29},{95, 93, 95, 93},{87, 85, 87, 85},
                {79, 77, 79, 77},{95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},
                {95, 93, 95, 93},{87, 85, 87, 85},{79, 77, 79, 77},{47, 45, 47, 45},
                {39, 37, 39, 37},{31, 29, 31, 29},{47, 45, 47, 45},{39, 37, 39, 37},
                {31, 29, 31, 29},{47, 45, 47, 45},{39, 37, 39, 37},{31, 29, 31, 29},
                };
            std::cout << *result << std::endl;
            COMPARE_TENSOR_DATA(result->mutable_data(), expected, 1e-6);

        }
    }


    static void test_OP() {
        {
            const Tensor<float> *a = new CPUTensor<float>({10, 10, 10},1);
            const Tensor<float> *b = new CPUTensor<float>({10, 10, 10},4);
            // MEASURE_TIME(const auto a = CPUTensor<float>({1000,1000,1000},2););
            // MEASURE_TIME(a - b);
            // MEASURE_TIME(*a + *b);


            // const auto a = CPUTensor<float>({2, 4, 1, 3},1);
            // const auto b = CPUTensor<float>({2, 1, 4, 3},4);
            // // MEASURE_TIME(const auto a = CPUTensor<float>({1000,1000,1000},2););
            // // MEASURE_TIME(a - b);
            // // MEASURE_TIME(a + b);
            // std::cout << a << std::endl;
            // const auto c = *b + *a;
            // std::cout << *c << std::endl;
        }

        // {
        //     auto a = CPUTensor<float>({20,20,20},2);
        //     const auto a_s = a.slice({"::2","::2","::2"});
        //     auto b = CPUTensor<float>({20,20,20},4);
        //     const auto b_s = b.slice({"::2","::2","::2"});
        //     // MEASURE_TIME(const auto a = CPUTensor<float>({1000,1000,1000},2););
        //     const auto c = *a_s + *b_s;
        //     // MEASURE_TIME(a + b);
        //     // MEASURE_TIME(*a_s + *b_s);
        //     std::cout << *c << std::endl;
        // }


        // {
        //     auto a = CPUTensor<float>({20,20,20},2);
        //     const auto a_s = a.slice({"::2","::2"});
        //     auto b = CPUTensor<float>({20,20,20},4);
        //     const auto b_s = b.slice({"::2","::2"});
        //     // MEASURE_TIME(const auto a = CPUTensor<float>({1000,1000,1000},2););
        //     const auto c = *a_s + *b_s;
        //     MEASURE_TIME(a + b);
        //     MEASURE_TIME(*a_s + *b_s);
        //     // std::cout << *c << std::endl;
        // }
    }

    static void test_slice() {
             // 测试用例1：2D张量的基本切片
        {
            const auto t1 = CPUTensor<float>::arange(0,30,1)->view({10,3})->slice({"1::8"});
            assert(t1->is_contiguous() == false);
            ASSERT_EQ_VECTOR(t1->get_strides(),std::vector<index_t>({24, 1}));
            ASSERT_EQ_VECTOR(t1->get_shape().dims(),std::vector<index_t>({2, 3}));
            std::vector<std::vector<float>> expected = {
                {3, 4, 5},
                {27, 28, 29}
            };
            std::cout << *t1 << std::endl;
            COMPARE_TENSOR_DATA(t1->contiguous()->data(), expected, 0);
        }

        // 测试用例2：2D张量的连续切片
        {
            const auto t1 = CPUTensor<float>::arange(0,9,1)->view({3,3})->slice({"1::2"});
            assert(t1->is_contiguous() == true);
            ASSERT_EQ_VECTOR(t1->get_shape().dims(),std::vector<index_t>({1, 3}));
            ASSERT_EQ_VECTOR(t1->get_strides(),std::vector<index_t>({3, 1}));
            std::vector<std::vector<float>> expected = {
                {3,4,5}
            };
            assert(t1->at({0,0}) == 3.0f && t1->at({0,1}) == 4.0f && t1->at({0,2}) == 5.0);
        }

        // 测试用例3：0维度标量
        {
            const auto t1 = CPUTensor<float>::scalar(3.14f);
            const auto t2 = t1->slice({});
            assert(t2->get_shape().dims().empty());
            assert(t2->mutable_data()[0] == 3.14f);
        }

        // 测试用例4：1维度向量的切片
        {
            const auto t1 = CPUTensor<float>::arange(0,10,1);
            const auto t2 = t1->slice({"2:7:2"});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({3}));
            std::vector<std::vector<float>> expected = {{2, 4, 6}};
            COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        // 测试用例5：2维度矩阵的复杂切片
        {
            const auto t1 = CPUTensor<float>::arange(0,25,1)->view({5,5});
            const auto t2 = t1->slice({"1:4:2", "::2"});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({2, 3}));
            std::vector<std::vector<float>> expected = {
                {5, 7, 9},
                {15, 17, 19}
            };
            COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        // 测试用例6：切片超出范围
        {
            const auto t1 = CPUTensor<float>::arange(0,10,1);
            const auto t2 = t1->slice({"5:15"});  // 超出范围的部分应该被忽略
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({5}));
            std::vector<std::vector<float>> expected = {{5, 6, 7, 8, 9}};
            COMPARE_TENSOR_DATA(t2->clone()->mutable_data(), expected, 0);
        }

        // 测试用例7：负数索引
        {
            const auto t1 = CPUTensor<float>::arange(0,10,1);
            const auto t2 = t1->slice({"-5:-1"});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({4}));
            std::vector<std::vector<float>> expected = {{5, 6, 7, 8}};
            COMPARE_TENSOR_DATA(t2->clone()->mutable_data(), expected, 0);
        }

        // 测试用例8：负数步长
        {
            const auto t1 = CPUTensor<float>::arange(0,10,1);
            const auto t2 = t1->slice({"::-1"});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({10}));
            std::vector<std::vector<float>> expected = {
                {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
            };
            COMPARE_TENSOR_DATA(t2->contiguous()->data(), expected, 0);
        }

        // 测试用例9：多维负数索引和步长
        {
            const auto t1 = CPUTensor<float>::arange(0,36,1)->view({6,6});
            const auto t2 = t1->slice({"-4:-1:2", "-3::2"});
            ASSERT_EQ_VECTOR(t2->get_shape().dims(),std::vector<index_t>({2, 2}));
            std::vector<std::vector<float>> expected = {
                {15, 17},
                {27, 29}
            };
            COMPARE_TENSOR_DATA(t2->contiguous()->mutable_data(), expected, 0);
        }

        std::cout << "All tests passed!" << std::endl;
    }
    // 运行所有测试
    static void run_all_tests() {
        // test_expand();
        test_OP();
        // test_slice();
        // test_repeat();
        // test_stack();
        // // test_randn();
        // test_flatten();
        // test_permute();
        // // test_omp();
        // test_cat();
        // test_unsqueeze();
        // test_squeeze();
        // test_view();
        // test_clone();
        // test_reshape();
        // test_non_contiguous();
        // test_transpose();
        std::cout << "All tests passed!" << std::endl;
    }
};


#endif //TEST_SQUEEZE_H
