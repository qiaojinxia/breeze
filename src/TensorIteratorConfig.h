#ifndef TENSORITERATORCONFIG_H
#define TENSORITERATORCONFIG_H
#include <vector>

#include "common/Macro.h"

namespace Breeze {

    template<typename... ScalarTypes>
    class TensorIterator;

    class TensorIteratorConfig {
    public:
        TensorIteratorConfig& set_enforce_safe_casting_to_output(bool enforce_safe_casting_to_output);
        TensorIteratorConfig& set_resize_outputs(bool resize_outputs);
        TensorIteratorConfig& set_check_all_same_dtype(bool check_all_same_dtype);
        TensorIteratorConfig& set_check_all_same_shape(bool check_all_same_shape);
        TensorIteratorConfig& set_reduce_dims(std::vector<index_t>& reduce_dims);
        TensorIteratorConfig& set_keep_keepdim(bool keepdim);
        TensorIteratorConfig& set_check_mem_overlap(bool check_mem_overla);
        TensorIteratorConfig& set_is_reduction(bool is_reduction);
        template<typename... ScalarTypes>
        TensorIterator<ScalarTypes...> build() const {
            return TensorIterator<ScalarTypes...>(*this);
        }
        static TensorIteratorConfig default_config();

    protected:
        template<typename... ScalarTypes>
        friend class TensorIterator;

    private:
        bool enforce_safe_casting_to_output_ = false;
        bool check_all_same_shape_ = false;
        bool resize_outputs_ = false;
        bool check_all_same_dtype_ = false;
        std::vector<index_t> reduce_dims_ = {};
        bool keep_keepdim_ = false;
        bool is_reduction_ = false;
        bool enforce_linear_iteration_ = false;
        bool check_mem_overlap_ = false;
    };

}

#endif //TENSORITERATORCONFIG_H