#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include "TensorStorage.h"
#include "CPUTensorOps.h"

namespace Breeze {

enum class Device {
    CPU,
    GPU
};

template<typename Dtype>
class Tensor {
protected:
    Device device;
    Shape shape;
    index_t offset_ = 0;  // Added offset_ as a member
    std::vector<index_t> strides_;  // Added strides_ as a member

    [[nodiscard]] bool is_contiguous_in_range(index_t start_dim, index_t end_dim) const;

public:
    static const CPUTensorOps<Dtype>* CpuOps;
    static const TensorOps<Dtype>* getOps();

    Tensor(Shape _shape, Device _device);
    virtual ~Tensor() = default;

    // Pure virtual operator overloads
    virtual Dtype operator[](const std::string& index) const = 0;
    virtual std::shared_ptr<Tensor> operator+(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator-(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator*(const Tensor& rhs) const = 0;
    virtual std::shared_ptr<Tensor> operator/(const Tensor& rhs) const = 0;

    virtual std::shared_ptr<Tensor> matmul(const Tensor& rhs) const = 0;

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

    virtual Dtype* mutable_data() = 0;
    virtual const Dtype* data() const = 0;
    [[nodiscard]] virtual index_t align_size() const = 0;
    virtual void set_initial_shape(Shape& shape) = 0;

    [[nodiscard]] virtual std::shared_ptr<Tensor> clone() const = 0;
    [[nodiscard]] virtual std::shared_ptr<Tensor> contiguous() = 0;
    [[nodiscard]] virtual const Dtype& at(const std::vector<index_t>& indices) const = 0;
    virtual void set_value(const std::vector<index_t>& indices, Dtype value) = 0;

    [[nodiscard]] virtual index_t size() const;
    virtual void to_cpu() = 0;
    virtual void to_gpu() = 0;
    virtual void print(std::ostream& os) const = 0;
    [[nodiscard]] virtual bool is_contiguous() const = 0;

    virtual void fill(Dtype value) = 0;
    virtual void fill(const std::function<Dtype(const std::vector<index_t>&)>& value_func) = 0;

    [[nodiscard]] const Shape& get_shape() const;
    [[nodiscard]] index_t get_offset() const;
    [[nodiscard]] Device get_device() const;
    [[nodiscard]] virtual std::vector<index_t> get_strides() const = 0;

    [[nodiscard]] index_t num_elements() const;
    [[nodiscard]] virtual index_t n_bytes() const = 0;

private:
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }
};

template<typename Dtype>
Tensor<Dtype>::Tensor(Shape _shape, const Device _device)
    : device(_device), shape(std::move(_shape)), offset_(0) , strides_(_shape.compute_strides()) {} // Initialize strides_

template<typename Dtype>
const TensorOps<Dtype>* Tensor<Dtype>::getOps() {
    if (CpuOps == nullptr) {
        const auto* _op = new CPUTensorOps<Dtype>();
        CpuOps = _op;
    }
    return CpuOps;
}

template<typename Dtype>
index_t Tensor<Dtype>::size() const {
    index_t store_size = 1;
    const auto& dims = shape.dims();
    const auto& strides = get_strides();

    for (index_t i = 0; i < dims.size(); ++i) {
        if (strides[i] != 0) {
            store_size *= dims[i];
        }
    }
    return store_size;
}

    template<typename Dtype>
    const Shape& Tensor<Dtype>::get_shape() const {
        return shape;
    }

    template<typename Dtype>
    index_t Tensor<Dtype>::get_offset() const {
        return offset_;
    }

    template<typename Dtype>
    Device Tensor<Dtype>::get_device() const {
        return device;
    }

    template<typename Dtype>
    index_t Tensor<Dtype>::num_elements() const {
        return shape.total_size();
    }

    template<typename Dtype>
    bool Tensor<Dtype>::is_contiguous_in_range(const index_t start_dim, index_t end_dim) const {
        const std::vector<index_t>& shape = get_shape().dims();
        const std::vector<index_t>& strides = get_strides();

        if (shape.empty() || (shape.size() == 1 && shape[0] == 0)) {
            return true;
        }

        if (end_dim == -1) {
            end_dim = shape.size() - 1;
        }

        if (start_dim > end_dim || end_dim >= shape.size()) {
            throw std::invalid_argument("Invalid dimension range");
        }
        index_t expected_stride = 1;

        for (index_t i = static_cast<index_t>(shape.size()) - 1; i >= 0; --i) {
            if (i <= end_dim && i >= start_dim && shape[i] != 1 && strides[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape[i];
        }
        return true;
    }

    template<typename Dtype>
    const CPUTensorOps<Dtype>* Tensor<Dtype>::CpuOps = nullptr;

} // namespace Breeze

#endif // TENSOR_H
