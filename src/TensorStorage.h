#ifndef TensorStorage_H
#define TensorStorage_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include "common/Utils.h"
#include "platform/SIMDFactory.h"
#include "./common/Macro.h"

namespace Breeze {
    struct CPUDevice {};
    struct GPUDevice {};

    template <typename ScalarType, typename DeviceType>
    class TensorStorage;

    template <typename ScalarType>
    class TensorStorage<ScalarType, CPUDevice> {
    public:
        explicit TensorStorage()
            : total_size_(0), padding_size_(0), data_(nullptr) {
        }

        explicit TensorStorage(const index_t size)
            : total_size_(size), padding_size_(0), data_(nullptr) {
            allocate_memory();
        }

        ~TensorStorage() {
            deallocate_memory();
        }

        // 禁用复制
        TensorStorage(const TensorStorage&) = delete;
        TensorStorage& operator=(const TensorStorage&) = delete;

        // 允许移动
        TensorStorage(TensorStorage&& other) noexcept
            : total_size_(other.total_size_), padding_size_(other.padding_size_), data_(other.data_) {
            other.data_ = nullptr;
            other.total_size_ = 0;
            other.padding_size_ = 0;
        }

        TensorStorage& operator=(TensorStorage&& other) noexcept {
            if (this != &other) {
                deallocate_memory();
                data_ = other.data_;
                total_size_ = other.total_size_;
                padding_size_ = other.padding_size_;
                other.data_ = nullptr;
                other.total_size_ = 0;
                other.padding_size_ = 0;
            }
            return *this;
        }

        void copy_to_device(const ScalarType* host_data, const index_t size) {
            if (size != total_size_) {
                throw std::runtime_error("Size mismatch in copy_to_device");
            }
            std::memcpy(data_, host_data, size * sizeof(ScalarType));
        }

        void copy_to_host(ScalarType* host_data, const index_t size) const {
            if (size != total_size_) {
                throw std::runtime_error("Size mismatch in copy_to_host");
            }
            std::memcpy(host_data, data_, size * sizeof(ScalarType));
        }

        ScalarType* data() { return data_; }
        const ScalarType* data() const { return data_; }

        [[nodiscard]] index_t total_size() const { return padding_size_; }
        [[nodiscard]] index_t total_bytes() const { return padding_size_ * sizeof(ScalarType); }

    private:
        void allocate_memory() {
            constexpr index_t alignment = 64;
            const index_t size = total_size_ * sizeof(ScalarType);
            const index_t padded_size = (size + alignment - 1) & ~(alignment - 1);
            data_ = static_cast<ScalarType*>(aligned_alloc(alignment, padded_size));
            if (data_ == nullptr) {
                throw std::bad_alloc();
            }
            padding_size_ = padded_size / sizeof(ScalarType);
        }

        void deallocate_memory() {
            if (data_ != nullptr) {
                free(data_);
                data_ = nullptr;
            }
        }

        index_t total_size_;
        index_t padding_size_;
        ScalarType* data_;
    };

    class Shape {
    public:
        explicit Shape(std::vector<index_t> dims) : dims_(std::move(dims)), strides_(compute_strides()) {}
        Shape(const std::initializer_list<index_t> dims) : dims_(dims), strides_(compute_strides()) {}
        Shape() : dims_({}), strides_({}) {}

        // 禁用复制
        Shape(const Shape&) = delete;
        Shape& operator=(const Shape&) = delete;

        // 允许移动
        Shape(Shape&& other) noexcept : dims_(std::move(other.dims_)), strides_(std::move(other.strides_)) {}

        Shape& operator=(Shape&& other) noexcept {
            if (this != &other) {
                dims_ = std::move(other.dims_);
                strides_ = std::move(other.strides_);
            }
            return *this;
        }

        [[nodiscard]] index_t dim(const index_t axis) const {
            if (axis >= ndim()) {
                throw std::out_of_range("Axis out of range");
            }
            return dims_[axis];
        }

        [[nodiscard]] index_t ndim() const {
            return static_cast<index_t>(dims_.size());
        }

        [[nodiscard]] index_t total_size() const {
            if (dims_.empty())
                return 1;
            return std::accumulate(dims_.begin(), dims_.end(), static_cast<index_t>(1), std::multiplies<>());
        }

        [[nodiscard]] const std::vector<index_t>& dims() const {
            return dims_;
        }

        [[nodiscard]] std::string dims_str() const {
            std::string s = "[";
            s += std::accumulate(std::begin(dims_), std::end(dims_), std::string{},
                [](const std::string& a, const index_t b) {
                    return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                });
            s += "]";
            return s;
        }

        [[nodiscard]] const std::vector<index_t>& strides() const {
            return strides_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
            os << '(';
            for (index_t i = 0; i < shape.ndim(); ++i) {
                os << shape.dims_[i];
                if (i != shape.ndim() - 1) {
                    os << ", ";
                }
            }
            os << ')';
            return os;
        }

        bool operator==(const Shape& other) const {
            return dims_ == other.dims_;
        }

        bool operator!=(const Shape& other) const {
            return !(*this == other);
        }

        [[nodiscard]] std::vector<index_t> compute_strides() const {
            std::vector<index_t> strides(dims_.size());
            if (!dims_.empty()) {
                strides.back() = 1;
                for (index_t i = static_cast<index_t>(dims_.size()) - 1; i > 0; --i) {
                    strides[i - 1] = strides[i] * dims_[i];
                }
            }
            return strides;
        }
    private:
        std::vector<index_t> dims_;
        std::vector<index_t> strides_;
    };
}

#endif // TensorStorage_H