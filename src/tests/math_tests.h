//
// Created by Apple on 2024/10/21.
//

#ifndef MATH_TESTS_H
#define MATH_TESTS_H
#include "CPUTensor.h"
#include <iostream>
#include <vector>

#include "../CPUTensor.h"
#include "../common/Macro.h"

using namespace Breeze;
static void test_Math()
{

    {

        const auto a = Tensor<float>::scalar(1);
        const auto b = Tensor<float>::arange(1,5,1);
        const auto c = *a / *b;
        const std::vector<std::vector<float>> expected = {
            {1.000000, 0.500000, 0.333333, 0.250000},
        };
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({4}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }


    {
        const auto a = Tensor<float>::scalar(2.0);
        const auto b = Tensor<float>::arange(1,50,1)->slice({"10:12"});
        const auto c = *a / *b;
        const std::vector<std::vector<float>> expected = {
            {0.181818, 0.166666},
        };
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }


    {
        const auto a = Tensor<float>::scalar(2.0);
        const auto b = Tensor<float>::arange(1,50,1)->slice({"20:25"});
        const auto c = *a / *b;
        const std::vector<std::vector<float>> expected = {
            {0.09523809, 0.0909090, 0.0869565, 0.08333333, 0.08},
        };
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(c);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({5}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(1,5,1);
        const std::vector<std::vector<float>> expected = {
            {0.000000, 0.693147, 1.098612, 1.386294},
        };
        auto b = a->log();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({4}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(100,200,1)->slice({"50:56"});
        const std::vector<std::vector<float>> expected = {
            {7.228819, 7.238405, 7.247928, 7.257388, 7.266787, 7.276124},
        };
        auto b = a->log2();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({6}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }


    {
        const auto a = Tensor<float>::arange(100,200,1)
            ->view({10,10})->slice({"5:","6:"});
        const std::vector<std::vector<float>> expected = {
            {2.193125, 2.195900, 2.198657, 2.201397},
            {2.220108, 2.222717, 2.225309, 2.227887},
            {2.245513, 2.247973, 2.250420, 2.252853},
            {2.269513, 2.271842, 2.274158, 2.276462},
            {2.292256, 2.294466, 2.296665, 2.298853},
        };
        auto b = a->log10();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({5,4}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }



    {
        const auto a = Tensor<float>::arange(1,5,1);
        const std::vector<std::vector<float>> expected = {
            {2.718282,  7.389056, 20.085537, 54.598150},
        };
        auto b = a->exp();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({4}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(0,50,1)
            ->view({2,5,5})->slice({"1:","2:",":"});
        const std::vector<std::vector<float>> expected = {
            {5.916080, 6.000000, 6.082763, 6.164414, 6.244998},
            {6.324555, 6.403124, 6.480741, 6.557438, 6.633250},
            {6.708204, 6.782330, 6.855655, 6.928203, 7.000000},
        };
        auto b = a->sqrt();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1, 3, 5}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(0,50,1)
            ->view({2,5,5})->slice({"1:","4:5","2:3"});
        const std::vector<std::vector<float>> expected = {
            {0.145865},
        };
        auto b = a->rsqrt();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1,1,1}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(0,50,1)
            ->view({2,5,5})->slice({":","2:5","1:3"});
        const std::vector<std::vector<float>> expected = {
            {0.301511, 0.288675},{0.250000, 0.242536},{0.218218, 0.213201},
            {0.166667, 0.164399},{0.156174, 0.154303},{0.147442, 0.145865}
        };
        auto b = a->rsqrt();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({2,3,2}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(0,60,1)
            ->view({3,4,5})->slice({"1:2","1:4","1:3"});
        const std::vector<std::vector<float>> expected = {
            {0.196116, 0.192450},{0.179605, 0.176777},{0.166667, 0.164399},
        };
        auto b = a->rsqrt();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({1,3,2}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }

    {
        const auto a = Tensor<float>::arange(-5,0,1);
        const std::vector<std::vector<float>> expected = {
            {5.,  4., 3. , 2., 1.},
        };
        auto b = a->abs();
        const auto tensor = std::dynamic_pointer_cast<Tensor<float>>(b);
        BREEZE_ASSERT(tensor->get_shape().dims() == std::vector<index_t>({5}));
        COMPARE_TENSOR_DATA(tensor->mutable_data(), expected, 1e-6);
    }
}
#endif //MATH_TESTS_H
