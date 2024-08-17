#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include "TensorStorage.h"
#include "TensorOps.h"

namespace Breeze {

enum class Device {
    CPU,
    GPU
};

template<typename T>
class Tensor {
protected:
    Device device;
    Shape shape;
    TensorOps<T> *ops;

public:
    Tensor(Shape _shape, Device _device,TensorOps<T>* _tensor_op);
    virtual ~Tensor() = default;

    virtual std::shared_ptr<Tensor> operator+(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator-(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator*(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator/(const Tensor& rhs) const = 0;

    virtual std::shared_ptr<Tensor> matmul(const Tensor& rhs) const = 0;



    [[nodiscard]] virtual std::shared_ptr<Tensor> reshape(const std::vector<int32_t>& new_shape) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> slice(const std::vector<std::tuple<int32_t, int32_t, int32_t>>& ranges) const  = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> view(const std::vector<int32_t>& new_shape) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> unsqueeze(int32_t dim) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> squeeze(int32_t dim) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> expand(const std::vector<int32_t>& new_shape) const = 0;

    virtual T* data() = 0;
    virtual const T* data() const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> clone() const = 0;
    [[nodiscard]] virtual const T& at(const std::vector<size_t>& indices) const = 0;
    virtual void set_value(const std::vector<size_t>& indices,T value) = 0;

    [[nodiscard]] virtual size_t size() const;
    virtual void to_cpu() = 0;
    virtual void to_gpu() = 0;
    virtual void print(std::ostream& os) const = 0;
    [[nodiscard]] virtual bool is_contiguous() const = 0;

    virtual void fill(T value) = 0;
    virtual void fill(const std::function<T(const std::vector<size_t>&)>& value_func) = 0;

    [[nodiscard]] const Shape& get_shape() const;
    [[nodiscard]] Device get_device() const;
    [[nodiscard]] virtual std::vector<size_t> get_strides() const = 0;

    [[nodiscard]] virtual std::vector<int32_t> get_steps() const = 0;
    [[nodiscard]] size_t num_elements() const;
private:

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }
};

    template<typename T>
    Tensor<T>::Tensor(Shape _shape,const  Device _device, TensorOps<T> *_tensor_op)
        :device(_device), shape(std::move(_shape)), ops(_tensor_op) {}

    template<typename T>
    size_t Tensor<T>::size() const {
        size_t store_size = 1;
        const auto& dims = shape.dims();
        const auto& strides = get_strides();

        for (size_t i = 0; i < dims.size(); ++i) {
            if ( strides[i] != 0 ) {
                store_size *= dims[i];
            }
        }
        return store_size;
    }

    template<typename T>
    const Shape& Tensor<T>::get_shape() const {
        return shape;
    }

    template<typename T>
    Device Tensor<T>::get_device() const {
        return device;
    }

    template<typename T>
    size_t Tensor<T>::num_elements() const {
        return shape.total_size();
    }

} // namespace Breeze

#endif // TENSOR_H