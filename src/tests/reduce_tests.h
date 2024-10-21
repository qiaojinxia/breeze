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

            // sum 测试标量
            {
                const auto a = Tensor<float>::scalar(1);
                const std::vector<std::vector<float>> expected = {
                    {1.}
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(a);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }
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

            // 非连续 sum
            {
                const auto a = Tensor<float>::arange(0,120,1)
                            ->view({2,3,4,5})->slice({":1","2:3","2:4",":"});;
                const auto b = a->sum({0,1});
                auto tensor = std::dynamic_pointer_cast<Tensor<float>>(a);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 1, 2, 5}));
                const std::vector<std::vector<float>> expected = {
                    {50., 51., 52., 53., 54.},
                    {55., 56., 57., 58., 59.},
                };
                tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2, 5}));
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
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2,3}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            // min vector
            {
                const auto a = Tensor<float>::vector("7, 3, 4, 5");
                const auto b = a->min({0});
                const std::vector<std::vector<float>> expected = {
                    {3},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
            }

            // min vector 非连续
            {
                const auto a = Tensor<float>::arange(0,30,1)
                            ->view({2,3,5})->slice({"0:1:1","2:3:1","1:5:1"});
                const auto b = a->min({2});
                const std::vector<std::vector<float>> expected = {
                    {11},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 1}));
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
                const auto a = Tensor<float>::scalar(3.3);
                const auto b = Tensor<float>::arange(0,30,1)
                    ->view({2,3,5})->slice({":1","1:3",":"});
                const auto c = *a * *b;
                const std::vector<std::vector<float>> expected = {
                    {16.5000, 19.8000, 23.1000, 26.4000, 29.7000},
                    {33.0000, 36.3000, 39.6000, 42.9000, 46.2000},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c->contiguous());
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1,2,5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

            {
                const auto a = Tensor<float>::arange(0,30,1)
                                    ->view({2,3,5})->slice({":1","1:3",":"});
                auto b = a->mean({1});
                auto c = *a  - *b;
                const std::vector<std::vector<float>> expected = {
                    {-2.5000, -2.5000, -2.5000, -2.5000, -2.5000},
                    {2.5000, 2.5000, 2.5000, 2.5000, 2.5000},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 2, 5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

            // 测试 std
            {
                const auto a = Tensor<float>::arange(0,30,1)
                                    ->view({2,3,5})->slice({":1","1:3",":"});
                const auto b = a->std({1});
                const std::vector<std::vector<float>> expected = {
                    {3.53553, 3.53553, 3.53553, 3.53553, 3.53553},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 5}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }
            // 测试 std
            {
                const auto a = Tensor<float>::arange(0,30,1)
                                            ->view({2,3,5})->slice({":1","1:3",":"});
                const auto b = a->std({2});
                const std::vector<std::vector<float>> expected = {
                    {1.58114, 1.58114},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 2}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

            // 测试 向量 var
            {
                const auto a = Tensor<float>::arange(0,30,1);
                const auto b = a->var({});
                const std::vector<std::vector<float>> expected = {
                    {77.5000},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

            // 测试 var
            {
                const auto a = Tensor<float>::arange(0,40,1)
                                                               ->view({5,8})->slice({"1:",":4"});
                const auto b = a->var({1});
                const std::vector<std::vector<float>> expected = {
                    {1.66667, 1.66667, 1.66667, 1.66667},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({4}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

            // 测试 var
            {
                const auto a = Tensor<float>::arange(0,30,1)
                                                    ->view({10,3});
                const auto b = a->var({0});
                const std::vector<std::vector<float>> expected = {
                    {82.5000, 82.5000, 82.5000},
                };
                const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
                BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({3}));
                COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-5);
            }

}
#endif //REDUCE_TEST_H
