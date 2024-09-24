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

    enum class TensorState {
        UNINITIALIZED,  // 未初始化
        ALLOCATED,      // 已分配内存但未设置数据
        INITIALIZED     // 已完全初始化（分配内存并设置数据）
    };
    class TensorBase {
    public:
        virtual ~TensorBase() = default;

        [[nodiscard]] virtual std::shared_ptr<TensorBase> view(const std::vector<index_t>& new_shape) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> reshape(const std::vector<index_t>& new_shape) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> slice(const std::vector<std::string>& range_strings) = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> unsqueeze(index_t dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> squeeze(index_t dim) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> squeeze() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> expand(const std::vector<index_t>& new_shape) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> transpose(index_t dim0, index_t dim1) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> permute(const std::vector<index_t>& dims) = 0;

        [[nodiscard]] virtual std::shared_ptr<TensorBase> flatten() = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> flatten(index_t start_dim, index_t end_dim) = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> repeat(const std::vector<index_t>& repeats) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> clone() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> contiguous() = 0;

        virtual void print(std::ostream& os) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> sin() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> cos() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> tan() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> atan() const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> pow(const TensorBase& other) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> operator+(const TensorBase& rhs) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> matmul(const TensorBase& rhs) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> operator-(const TensorBase& rhs) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> operator*(const TensorBase& rhs) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> operator/(const TensorBase& rhs) const = 0;
        [[nodiscard]] virtual std::shared_ptr<TensorBase> sum(std::vector<index_t> dims) = 0;
        virtual void operator+=(const TensorBase& rhs) = 0;
        virtual void operator-=(const TensorBase& rhs) = 0;
        virtual void operator*=(const TensorBase& rhs) = 0;
        virtual void operator/=(const TensorBase& rhs) = 0;

        [[nodiscard]] TensorState get_state() const { return state_; }
        void set_state(const TensorState new_state) { state_ = new_state; }

        [[nodiscard]] bool is_initialized() const { return state_ == TensorState::INITIALIZED; }
        [[nodiscard]] bool is_allocated() const { return state_ == TensorState::ALLOCATED || state_ == TensorState::INITIALIZED;}
        [[nodiscard]] bool is_uninitialized() const { return state_ == TensorState::UNINITIALIZED; }

    private:
        TensorState state_ = TensorState::UNINITIALIZED;
        friend std::ostream& operator<<(std::ostream& os, const TensorBase& tensor) {
            tensor.print(os);
            return os;
        }
    };

    template<typename ScalarType>
    class CPUTensor;

    template<typename ScalarType>
    class Tensor : public TensorBase {
    protected:
        Device device;
        Shape shape;
        index_t offset_ = 0;
        std::vector<index_t> strides_;
        [[nodiscard]] bool is_contiguous_in_range(index_t start_dim, index_t end_dim) const;

    public:
        Tensor(Shape _shape, Device _device, TensorState state);
        ~Tensor() override = default;

        // Pure virtual methods
        virtual ScalarType operator[](const std::string& index) const = 0;

        virtual ScalarType* mutable_data() = 0;
        virtual const ScalarType* data() const = 0;
        [[nodiscard]] virtual index_t align_size() const = 0;
        virtual void set_initial_shape(Shape& shape) = 0;


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
        static std::shared_ptr<Tensor> create_tensor(std::vector<index_t> shape,Device device = Device::CPU);
        static std::shared_ptr<Tensor> create_tensor(std::vector<index_t> shape, ScalarType value,Device device = Device::CPU);
        static std::shared_ptr<Tensor> arange(ScalarType start, ScalarType end, ScalarType step,  Device device= Device::CPU);
        static std::shared_ptr<Tensor> vector(index_t size, Device device= Device::CPU);
        static std::shared_ptr<Tensor> scalar(ScalarType value, Device device= Device::CPU);
    };

    template<typename ScalarType>
    std::shared_ptr<Tensor<ScalarType>> Tensor<ScalarType>::create_tensor(std::vector<index_t> shape, const  Device device) {
        if (device == Device::CPU) {
            return std::make_shared<CPUTensor<ScalarType>>(shape);
        } else {
            return nullptr;
        }
    }

    template<typename ScalarType>
    std::shared_ptr<Tensor<ScalarType>> Tensor<ScalarType>::create_tensor(std::vector<index_t> shape,
        ScalarType value, const Device device) {
        if (device == Device::CPU) {
            return std::make_shared<CPUTensor<ScalarType>>(Shape(std::move(shape)), value);
        } else {
            return nullptr;
        }
    }

    template<typename ScalarType>
    std::shared_ptr<Tensor<ScalarType>> Tensor<ScalarType>::arange(ScalarType start, ScalarType end, ScalarType step, const Device device) {
        if (device == Device::CPU) {
            return CPUTensor<ScalarType>::arange(start, end, step);
        } else {
            return nullptr;
        }
    }

    template<typename ScalarType>
    std::shared_ptr<Tensor<ScalarType>> Tensor<ScalarType>::vector(const index_t size, const Device device) {
        if (device == Device::CPU) {
            return CPUTensor<ScalarType>::vector(size);
        } else {
            return nullptr;
        }
    }


    template<typename ScalarType>
    std::shared_ptr<Tensor<ScalarType>> Tensor<ScalarType>::scalar(ScalarType value, const Device device) {
        if (device == Device::CPU) {
            return CPUTensor<ScalarType>::scalar(value);
        } else {
            return nullptr;
        }
    }


    // Implementation part remains unchanged
    template<typename ScalarType>
    Tensor<ScalarType>::Tensor(Shape _shape, const Device _device, const TensorState state)
        : device(_device), shape(std::move(_shape)), strides_(this->get_shape().compute_strides()) {
            this->set_state(state);
        }

    template<typename ScalarType>
    index_t Tensor<ScalarType>::size() const {
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

    template<typename ScalarType>
    const Shape& Tensor<ScalarType>::get_shape() const {
        return shape;
    }

    template<typename ScalarType>
    index_t Tensor<ScalarType>::get_offset() const {
        return offset_;
    }

    template<typename ScalarType>
    Device Tensor<ScalarType>::get_device() const {
        return device;
    }

    template<typename ScalarType>
    index_t Tensor<ScalarType>::num_elements() const {
        return shape.total_size();
    }

    template<typename ScalarType>
    bool Tensor<ScalarType>::is_contiguous_in_range(const index_t start_dim, index_t end_dim) const {
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