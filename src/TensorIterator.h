#ifndef TENSORITERATOR_H
#define TENSORITERATOR_H

#include "Tensor.h"
#include <vector>
#include <omp.h>
#include <iostream>

namespace Breeze {
    template<typename T>
    class Tensor;

    template<typename T>
    class CPUTensor;

    template<typename T>
    class TensorIterator {
    public:
        struct OperandInfo {
            const T* data;
            std::vector<size_t> strides;
            std::vector<int32_t> steps;
        };

        TensorIterator(Tensor<T>& result, const Tensor<T>& a, const Tensor<T>& b)
            : result_(result), a_(a), b_(b), operands_() {
            setup();
        }

        void setup() {
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

        static std::unique_ptr<TensorIterator> binary_op(Tensor<T>& result, const Tensor<T>& a, const Tensor<T>& b) {
            return std::make_unique<TensorIterator>(result, a, b);
        }

        template<typename Func>
        void for_each(Func func) {
            if (shape_.empty()) {
                call_func(func, std::index_sequence_for<OperandInfo, OperandInfo, OperandInfo>{});
                return;
            }

            size_t outer_dim = 1;
            for (size_t i = 0; i < shape_.size() - 1; ++i) {
                outer_dim *= shape_[i];
            }
            const size_t inner_dim = shape_.back();
#pragma omp parallel for
            for (size_t i = 0; i < outer_dim; ++i) {
                std::vector<size_t> coords(shape_.size() - 1);
                size_t temp = i;
                for (int k = static_cast<int>(shape_.size()) - 2; k >= 0; --k) {
                    coords[k] = temp % shape_[k];
                    temp /= shape_[k];
                }

                std::vector<size_t> offsets(operands_.size(), 0);
                for (size_t op = 0; op < operands_.size(); ++op) {
                    for (size_t k = 0; k < coords.size(); ++k) {
                        offsets[op] += coords[k] * operands_[op].strides[k] * operands_[op].steps[k];
                    }
                }

                std::vector<size_t> incs(operands_.size());
                for (size_t op = 0; op < operands_.size(); ++op) {
                    incs[op] = operands_[op].strides.back() * operands_[op].steps.back();
                }

                call_func(func, std::index_sequence_for<OperandInfo, OperandInfo, OperandInfo>{},
                          inner_dim, offsets, incs);
            }
        }

    private:
        Tensor<T>& result_;
        const Tensor<T>& a_;
        const Tensor<T>& b_;
        std::vector<size_t> shape_;
        std::vector<OperandInfo> operands_;

        template<typename Func, size_t... Is>
        void call_func(Func& func, std::index_sequence<Is...>) {
            try_call_func<Func, Is...>(func);
        }

        template<typename Func, size_t... Is>
        void call_func(Func& func, std::index_sequence<Is...>,
                       const size_t inner_dim, const std::vector<size_t>& offsets,
                       const std::vector<size_t>& incs) {
            try_call_func<Func, Is...>(func, inner_dim, offsets, incs);
        }

        template<typename Func, size_t... Is>
        void try_call_func(Func& func) {
            if constexpr (std::is_invocable_v<Func, T*, const T*, const T*>) {
                func(const_cast<T*>(operands_[Is].data)...);
            } else if constexpr (std::is_invocable_v<Func, T*, const T*, const T*, size_t>) {
                func(const_cast<T*>(operands_[Is].data)..., 1);
            } else if constexpr (std::is_invocable_v<Func, T*, const T*, const T*, size_t, size_t, size_t, size_t>) {
                func(const_cast<T*>(operands_[Is].data)..., 1, 1, 1, 1);
            }
        }

        template<typename Func, size_t... Is>
        void try_call_func(Func& func, size_t inner_dim, const std::vector<size_t>& offsets,
                           const std::vector<size_t>& incs) {
            if constexpr (std::is_invocable_v<Func, T*, const T*, const T*>) {
                func(const_cast<T*>(operands_[Is].data) + offsets[Is]...);
            } else if constexpr (std::is_invocable_v<Func, T*, const T*, const T*, size_t>) {
                func(const_cast<T*>(operands_[Is].data) + offsets[Is]..., inner_dim);
            } else if constexpr (std::is_invocable_v<Func, T*, const T*, const T*, size_t, size_t, size_t, size_t>) {
                func(const_cast<T*>(operands_[Is].data) + offsets[Is]..., inner_dim, incs[Is]...);
            }
        }
    };
}

#endif //TENSORITERATOR_H