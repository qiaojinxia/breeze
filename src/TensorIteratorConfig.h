#ifndef TENSORITERATORCONFIG_H
#define TENSORITERATORCONFIG_H

namespace Breeze {

    template<typename... ScalarTypes>
    class TensorIterator;

    class TensorIteratorConfig {
    public:
        TensorIteratorConfig& set_enforce_safe_casting_to_output(bool enforce_safe_casting_to_output);
        TensorIteratorConfig& set_resize_outputs(bool resize_outputs);
        TensorIteratorConfig& set_check_all_same_dtype(bool check_all_same_dtype);
        TensorIteratorConfig& set_check_all_same_shape(bool check_all_same_shape);

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
    };

}

#endif //TENSORITERATORCONFIG_H