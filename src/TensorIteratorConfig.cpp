//
// Created by mac on 2024/9/19.
//

#include "TensorIteratorConfig.h"
namespace Breeze {
    TensorIteratorConfig& TensorIteratorConfig::set_enforce_safe_casting_to_output(const bool enforce_safe_casting_to_output) {
        enforce_safe_casting_to_output_ = enforce_safe_casting_to_output;
        return *this;
    }

    TensorIteratorConfig& TensorIteratorConfig::set_resize_outputs(const bool resize_outputs) {
        resize_outputs_ = resize_outputs;
        return *this;
    }

    TensorIteratorConfig& TensorIteratorConfig::set_check_all_same_shape(const bool check_all_same_shape) {
        check_all_same_shape_ = check_all_same_shape;
        return *this;
    }

    TensorIteratorConfig & TensorIteratorConfig::set_reduce_dims(std::vector<index_t> &reduce_dims) {
        reduce_dims_ = std::move(reduce_dims);
        return *this;
    }

    TensorIteratorConfig & TensorIteratorConfig::set_keep_keep_dim(const bool keep_dim) {
        keep_keep_dim_ = keep_dim;
        return *this;
    }

    TensorIteratorConfig & TensorIteratorConfig::set_check_mem_overlap(const bool check_mem_overlap) {
        check_mem_overlap_ = check_mem_overlap;
        return *this;
    }

    TensorIteratorConfig & TensorIteratorConfig::set_is_reduction(bool is_reduction) {
        is_reduction_ = is_reduction;
        return *this;
    }


    TensorIteratorConfig& TensorIteratorConfig::set_check_all_same_dtype(const bool check_all_same_dtype) {
        check_all_same_dtype_ = check_all_same_dtype;
        return *this;
    }

    TensorIteratorConfig TensorIteratorConfig::default_config(){
        TensorIteratorConfig config;
        config
            .set_enforce_safe_casting_to_output(false)
            .set_resize_outputs(false)
            .set_check_all_same_shape(false)
            .set_check_all_same_dtype(false);
        return config;
    }

}
