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

        // 禁用复制
        TensorStorage(const TensorStorage&) = delete;
        TensorStorage& operator=(const TensorStorage&) = delete;

        // 允许移动
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

        void copyToDevice(const T* host_data, const size_t size) {
            if (size != total_size) {
                throw std::runtime_error("Size mismatch in copyToDevice");
            }
            std::memcpy(data, host_data, size * sizeof(T));
        }

        void copyToHost(T* host_data, const size_t size) const {
            if (size != total_size) {
                throw std::runtime_error("Size mismatch in copyToHost");
            }
            std::memcpy(host_data, data, size * sizeof(T));
        }

        T* getData() { return data; }
        const T* getData() const { return data; }
        [[nodiscard]] size_t getTotalSize() const { return total_size; }
        [[nodiscard]] size_t getTotalBytes() const { return total_size * sizeof(T); }
    private:
        void allocateMemory() {
            constexpr size_t alignment = 64;
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
            // 将 '[' 作为起始字符
            std::string s = "[";  // 用双引号表示字符串

            // 使用 std::accumulate 拼接字符串
            s += std::accumulate(std::begin(dims_), std::end(dims_), std::string{},
                [](const std::string& a,const size_t b) {
                    // 如果 a 为空则返回第一个元素 b 否则添加 ',' 和 b
                    return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                });

            // 添加 ']' 作为结束字符
            s += "]";  // 用双引号表示字符串

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

#endif //TensorStorage_H