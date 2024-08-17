#ifndef TensorStorage_H
#define TensorStorage_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include "common/Utils.h"

namespace Breeze {
    struct CPUDevice {};
    struct GPUDevice {};

    template <typename T, typename DeviceType>
    class TensorStorage;

    template <typename T>
    class TensorStorage<T, CPUDevice> {
    public:
        explicit TensorStorage(const size_t size)
        : total_size(size), data(nullptr) {
            allocateMemory();
        }

        ~TensorStorage() {
            deallocateMemory();
        }

        // 禁用复制构造函数和赋值运算符
        TensorStorage(const TensorStorage&) = delete;
        TensorStorage& operator=(const TensorStorage&) = delete;

        // 移动构造函数和移动赋值运算符
        TensorStorage(TensorStorage&& other) noexcept
            : total_size(other.total_size), data(other.data) {
            other.data = nullptr;
            other.total_size = 0;
        }

        TensorStorage& operator=(TensorStorage&& other) noexcept {
            if (this != &other) {
                deallocateMemory();
                data = other.data;
                total_size = other.total_size;
                other.data = nullptr;
                other.total_size = 0;
            }
            return *this;
        }

        void copyToDevice(const T* host_data,const size_t size) {
            if (size != total_size) {
                throw std::runtime_error("Size mismatch in copyToDevice");
            }
            std::memcpy(data, host_data, size * sizeof(T));
        }

        void copyToHost(T* host_data,const size_t size) const {
            if (size != total_size) {
                throw std::runtime_error("Size mismatch in copyToHost");
            }
            std::memcpy(host_data, data, size * sizeof(T));
        }

        T* getData() { return data; }
        const T* getData() const { return data; }
        [[nodiscard]] size_t getTotalSize() const { return total_size; }

    private:
        void allocateMemory() {
            // 使用 aligned_alloc 来确保内存对齐
            constexpr size_t alignment = 64;  // 假设使用 64 字节对齐（适用于 AVX-512）
            const size_t size = total_size * sizeof(T);
            const size_t padded_size = (size + alignment - 1) & ~(alignment - 1);
            data = static_cast<T*>(aligned_alloc(alignment, padded_size));
            if (data == nullptr) {
                throw std::bad_alloc();
            }
        }

        void deallocateMemory() {
            if (data != nullptr) {
                free(data);
                data = nullptr;
            }
        }

        size_t total_size;
        T* data;
    };

    class Shape {
    public:
        explicit Shape(const std::vector<size_t>& dims) : dims_(dims) {}

        Shape(const std::initializer_list<size_t> dims) : dims_(dims) {}

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
            //标空为量 也返回 1 个
            if (dims_.empty())
                return 1;
            return std::accumulate(dims_.begin(), dims_.end(), 1ULL, std::multiplies<>());
        }

        [[nodiscard]] std::vector<size_t> dims() const {
            return dims_;
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

#endif //TensorStorage_H