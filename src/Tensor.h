#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <memory>
#include <functional>
#include "TensorStorage.h"
#include "ScalarType.h"
namespace Breeze {

enum class Device {
    CPU,
    GPU
};

class ScalarBase {
    public:
        virtual ~ScalarBase() = default;
        virtual void print(std::ostream& os) const = 0;
};

class TensorBase {
public:
    virtual ~TensorBase() = default;
    virtual void print(std::ostream& os) const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> sin() const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> cos() const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> tan() const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> atan() const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> pow(const TensorBase& other) const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> operator+(const TensorBase& rhs) const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> operator-(const TensorBase& rhs) const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> operator*(const TensorBase& rhs) const = 0;
    [[nodiscard]] virtual std::shared_ptr<TensorBase> operator/(const TensorBase& rhs) const = 0;
    virtual void operator+=(const TensorBase& rhs) = 0;
    virtual void operator-=(const TensorBase& rhs) = 0;
    virtual void operator*=(const TensorBase& rhs) = 0;
    virtual void operator/=(const TensorBase& rhs) = 0;

private:
    friend std::ostream& operator<<(std::ostream& os, const TensorBase& tensor) {
        tensor.print(os);
        return os;
    }
};

template<typename ScalarType>
class Tensor : public TensorBase {
protected:
    Device device;
    Shape shape;
    index_t offset_ = 0;
    std::vector<index_t> strides_;
    [[nodiscard]] bool is_contiguous_in_range(index_t start_dim, index_t end_dim) const;

public:
    Tensor(Shape _shape, Device _device);
    ~Tensor() override = default;


    // Pure virtual methods
    virtual ScalarType operator[](const std::string& index) const = 0;

    [[nodiscard]] virtual std::shared_ptr<Tensor> reshape(const std::vector<index_t>& new_shape) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> slice(const std::vector<std::string>& range_strings) = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> view(const std::vector<index_t>& new_shape) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> unsqueeze(index_t dim) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> squeeze(index_t dim) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> squeeze() const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> expand(const std::vector<index_t>& new_shape) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> transpose(index_t dim0, index_t dim1) const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> permute(const std::vector<index_t>& dims) = 0;

    [[nodiscard]] virtual std::shared_ptr<Tensor> flatten() = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> flatten(index_t start_dim, index_t end_dim) = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> repeat(const std::vector<index_t>& repeats) const = 0;

    virtual ScalarType* mutable_data() = 0;
    virtual const ScalarType* data() const = 0;
    [[nodiscard]] virtual index_t align_size() const = 0;
    virtual void set_initial_shape(Shape& shape) = 0;

    [[nodiscard]] virtual std::shared_ptr<Tensor> clone() const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> contiguous() = 0;
    [[nodiscard]] virtual const ScalarType& at(const std::vector<index_t>& indices) const = 0;
    virtual void set_value(const std::vector<index_t>& indices, ScalarType value) = 0;

    [[nodiscard]] virtual index_t size() const;
    virtual void to_cpu() = 0;
    virtual void to_gpu() = 0;
    void print(std::ostream& os) const override = 0;
    [[nodiscard]] virtual bool is_contiguous() const = 0;

    virtual void fill(ScalarType value) = 0;
    virtual void fill(const std::function<ScalarType(const std::vector<index_t>&)>& value_func) = 0;

    [[nodiscard]] const Shape& get_shape() const;
    [[nodiscard]] index_t get_offset() const;
    [[nodiscard]] Device get_device() const;
    [[nodiscard]] virtual std::vector<index_t> get_strides() const = 0;

    [[nodiscard]] index_t num_elements() const;
    [[nodiscard]] virtual index_t n_bytes() const = 0;
};

// Implementation part remains unchanged
template<typename scalar_t>
Tensor<scalar_t>::Tensor(Shape _shape, const Device _device)
    : device(_device), shape(std::move(_shape)), strides_(this->get_shape().compute_strides()) {}

template<typename scalar_t>
index_t Tensor<scalar_t>::size() const {
    index_t store_size = 1;
    const auto& dims = shape.dims();
    const auto& strides = get_strides();

    for (size_t i = 0; i < dims.size(); ++i) {
        if (strides[i] != 0) {
            store_size *= dims[i];
        }
    }
    return store_size;
}

template<typename scalar_t>
const Shape& Tensor<scalar_t>::get_shape() const {
    return shape;
}

template<typename scalar_t>
index_t Tensor<scalar_t>::get_offset() const {
    return offset_;
}

template<typename scalar_t>
Device Tensor<scalar_t>::get_device() const {
    return device;
}

template<typename scalar_t>
index_t Tensor<scalar_t>::num_elements() const {
    return shape.total_size();
}

template<typename scalar_t>
bool Tensor<scalar_t>::is_contiguous_in_range(const index_t start_dim, index_t end_dim) const {
    const std::vector<index_t>& shape = get_shape().dims();
    const std::vector<index_t>& strides = get_strides();
    const auto ndim = get_shape().ndim();

    if (shape.empty() || (ndim == 1 && shape[0] == 0)) {
        return true;
    }

    if (end_dim == -1) {
        end_dim = ndim - 1;
    }

    if (start_dim > end_dim || end_dim >= ndim) {
        throw std::invalid_argument("Invalid dimension range");
    }

    index_t expected_stride = 1;
    for (index_t i = ndim - 1; i >= 0; --i) {
        if (i <= end_dim && i >= start_dim && shape[i] != 1 && strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

} // namespace Breeze

#endif // TENSOR_H