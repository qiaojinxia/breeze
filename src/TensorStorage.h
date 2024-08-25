#ifndef TensorStorage_H
#define TensorStorage_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include "common/Utils.h"
#include "platform/SIMDFactory.h"

namespace Breeze {

    struct CPUDevice {};
    struct GPUDevice {};

    template <typename T, typename DeviceType>
    class TensorStorage;

    template <typename T>
    class TensorStorage<T, CPUDevice> {
    public:
        explicit TensorStorage(const size_t size)
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

        void copy_to_device(const T* host_data, const size_t size) {
            if (size != total_size_) {
                throw std::runtime_error("Size mismatch in copy_to_device");
            }
            std::memcpy(data_, host_data, size * sizeof(T));
        }

        void copy_to_host(T* host_data, const size_t size) const {
            if (size != total_size_) {
                throw std::runtime_error("Size mismatch in copy_to_host");
            }
            std::memcpy(host_data, data_, size * sizeof(T));
        }

        T* data() { return data_; }
        const T* data() const { return data_; }

        [[nodiscard]] size_t total_size() const { return padding_size_; }
        [[nodiscard]] size_t total_bytes() const { return padding_size_ * sizeof(T); }

    private:
        void allocate_memory() {
            constexpr size_t alignment = 64;
            const size_t size = total_size_ * sizeof(T);
            const size_t padded_size = (size + alignment - 1) & ~(alignment - 1);
            data_ = static_cast<T*>(aligned_alloc(alignment, padded_size));
            if (data_ == nullptr) {
                throw std::bad_alloc();
            }
            padding_size_ = padded_size / sizeof(T);
        }

        void deallocate_memory() {
            if (data_ != nullptr) {
                free(data_);
                data_ = nullptr;
            }
        }

        size_t total_size_;
        size_t padding_size_;
        T* data_;
    };

    class Shape {
    public:
        explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}
        Shape(const std::initializer_list<size_t> dims) : dims_(dims) {}
        Shape() : dims_({}) {}

        // 禁用复制
        Shape(const Shape&) = delete;
        Shape& operator=(const Shape&) = delete;

        // 允许移动
        Shape(Shape&& other) noexcept : dims_(std::move(other.dims_)) {}

        Shape& operator=(Shape&& other) noexcept {
            if (this != &other) {
                dims_ = std::move(other.dims_);
            }
            return *this;
        }

        [[nodiscard]] size_t dim(const size_t axis) const {
            if (axis >= dims_.size()) {
                throw std::out_of_range("Axis out of range");
            }
            return dims_[axis];
        }

        [[nodiscard]] size_t ndim() const {
            return dims_.size();
        }

        [[nodiscard]] size_t total_size() const {
            if (dims_.empty())
                return 1;
            return std::accumulate(dims_.begin(), dims_.end(), 1ULL, std::multiplies<>());
        }

        [[nodiscard]] const std::vector<size_t>& dims() const {
            return dims_;
        }

        [[nodiscard]] std::string dims_str() const {
            std::string s = "[";
            s += std::accumulate(std::begin(dims_), std::end(dims_), std::string{},
                [](const std::string& a, const size_t b) {
                    return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                });
            s += "]";
            return s;
        }

        [[nodiscard]] std::vector<size_t> strides() const {
            return Utils::compute_strides(this->dims());
        }

        friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
            os << '(';
            for (size_t i = 0; i < shape.dims_.size(); ++i) {
                os << shape.dims_[i];
                if (i != shape.dims_.size() - 1) {
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

    private:
        std::vector<size_t> dims_;
    };
}

#endif // TensorStorage_H
