//
// Created by caomaobay on 2024/8/23.
//

#include "TensorIterator.h"
namespace Breeze {
    template<typename T>
    TensorIterator<T>::TensorIterator(Tensor<T>& result, const Tensor<T>& a, const Tensor<T>& b)
    : result_(result), a_(a), b_(b), operands_() {
        setup();
    }

    template<typename T>
    void TensorIterator<T>::setup(){
        auto [a_strides, b_strides, target_shape] =
            Utils::calc_broadcast_shape(a_.get_shape().dims(), b_.get_shape().dims(), false);
        auto shape = Shape(std::vector<size_t>(target_shape.begin(), target_shape.end()));
        result_.set_initial_shape(shape);
        shape_ = target_shape;
        operands_.clear();
        operands_.push_back(OperandInfo{result_.data(), result_.get_strides(), result_.get_steps()});
        operands_.push_back(OperandInfo{a_.data(), a_strides, a_.get_steps()});
        operands_.push_back(OperandInfo{b_.data(), b_strides, b_.get_steps()});
    }

    template<typename T>
    TensorIterator<T> TensorIterator<T>::binary_op(Tensor<T>& result, const Tensor<T>& a, const Tensor<T>& b){
        auto op = TensorIterator(result, a, b);
        return op;
    }


    template class TensorIterator<float>;
    template class TensorIterator<double>;
}