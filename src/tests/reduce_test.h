//
// Created by mac on 2024/10/10.
//

#ifndef REDUCE_TEST_H
#define REDUCE_TEST_H
#include "CPUTensor.h"
#include <iostream>
#include <vector>

#include "../CPUTensor.h"
#include "../common/Macro.h"

using namespace Breeze;
static void test_Reduce() {
            // sum 测试案例
            {
                const auto a = Tensor<float>::arange(0,120,1)->view({2,3,4,5});
                const auto b = a->sum({0,1,2}, true);
                const std::vector<std::vector<float>> expected = {
                    {1380., 1404., 1428., 1452., 1476.}
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1,1,1,5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                const auto a = Tensor<float>::arange(0,120,1)->view({2,3,4,5});
                const auto b = a->sum({0,2});
                const std::vector<std::vector<float>> expected = {
                    {300., 308., 316., 324., 332.},
                    {460., 468., 476., 484., 492.},
                    {620., 628., 636., 644., 652.},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({3, 5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                const auto a = Tensor<float>::arange(0,120,1)->view({2,3,4,5});
                const auto b = a->sum({0,1});
                const std::vector<std::vector<float>> expected = {
                    {300., 306., 312., 318., 324.},
                    {330., 336., 342., 348., 354.},
                    {360., 366., 372., 378., 384.},
                    {390., 396., 402., 408., 414.},

                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({4, 5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                const auto a = Tensor<float>::arange(0,120,1)->view({2,3,4,5});
                const auto b = a->sum({1,3}, true);
                const std::vector<std::vector<float>> expected = {
                    {330.,  405.,  480.,  555.},
                    {1230., 1305., 1380., 1455.},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2,1,4,1}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            // max 测试案例

             {
                const auto a = Tensor<float>::arange(0,30,1)->view({2,3,5});
                const auto b = a->max({0});
                const std::vector<std::vector<float>> expected = {
                    {15., 16., 17., 18., 19.},
                    {20., 21., 22., 23., 24.},
                    {25., 26., 27., 28., 29.},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({3,5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                        const auto a = Tensor<float>::arange(0,30,1)->view({2,3,5});
                        const auto b = a->max({1});
                        const std::vector<std::vector<float>> expected = {
                            {10., 11., 12., 13., 14.},
                            {25., 26., 27., 28., 29.},
                        };
                        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2,5}));
                        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            // min 测试案例

            {
                        const auto a = Tensor<float>::arange(0,30,1)->view({2,3,5});
                        const auto b = a->min({2});
                        const std::vector<std::vector<float>> expected = {
                            {0., 5., 10., 15., 20., 25.},
                        };
                        std::cout << *b << std::endl;
                        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2,3}));
                        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                const auto a = Tensor<float>::scalar(1);
                const auto b = Tensor<float>::create_tensor({2,5,3},0.3);
                const auto c = *a * *b;
                const std::vector<std::vector<float>> expected = {
                    {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3},
                    {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3},
                    {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            {
                const auto tensor1 = Tensor<float>::create_tensor({1,1,1},0.1);
                const auto tensor2 = Tensor<float>::create_tensor({1,1,1},0.3);
                const std::vector<std::vector<float>> expected = {
                    {0.03},
                };
                auto c = *tensor1 * *tensor2;
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

}
#endif //REDUCE_TEST_H
