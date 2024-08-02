//
// Created by mac on 2024/7/31.
//
#include <random>
#include "AlignedDataBlob.h"
namespace Breeze {
template<typename T>
AlignedDataBlob<T>::AlignedDataBlob(const std::vector<size_t>& shape) {
    total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
    auto [buf, ptr] = allocate_aligned(total_size, 64);
    this->shape = shape;
    buffer = std::move(buf);
    data = ptr;
}

template<typename T>
T& AlignedDataBlob<T>::at(const std::vector<size_t>& indices) {
    return data[calculateIndex(indices)];
}

template<typename T>
const T& AlignedDataBlob<T>::at(const std::vector<size_t>& indices) const {
    return data[calculateIndex(indices)];
}

template<typename T>
AlignedDataBlob<T>::AlignedDataBlob(AlignedDataBlob&& other) noexcept
    : data(other.data), buffer(std::move(other.buffer)), shape(std::move(other.shape)), total_size(other.total_size) {
    other.data = nullptr;
    other.total_size = 0;
}

template<typename T>
AlignedDataBlob<T>& AlignedDataBlob<T>::operator=(AlignedDataBlob&& other) noexcept {
    if (this != &other) {
        buffer = std::move(other.buffer);
        data = other.data;
        shape = std::move(other.shape);
        total_size = other.total_size;
        other.data = nullptr;
        other.total_size = 0;
    }
    return *this;
}

template<typename T>
size_t AlignedDataBlob<T>::calculateIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Invalid number of indices");
    }
    size_t index = 0;
    size_t multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return index;
}

template<typename T>
void AlignedDataBlob<T>::reshape(const std::vector<size_t>& new_shape) {
    const size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
    if (new_total_size != total_size) {
        throw std::invalid_argument("New shape does not match total size");
    }
    shape = new_shape;
}

template<typename T>
   void AlignedDataBlob<T>::fillRandom(T minValue, T maxValue) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minValue, maxValue);
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = static_cast<T>(dis(gen));
    }
}

template<typename T>
std::pair<std::unique_ptr<T[]>, T*> AlignedDataBlob<T>::allocate_aligned(const size_t size,const size_t alignment) {
    // Ensure alignment is a power of two
    if ((alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of two");
    }

    // Allocate buffer with enough space for alignment
    std::unique_ptr<T[]> buffer(new T[size + alignment - 1]);

    // Get a void pointer to the buffer
    void* ptr = buffer.get();
    std::size_t space = (size + alignment - 1) * sizeof(T);

    // Align the pointer
    if (std::align(alignment, size * sizeof(T), ptr, space)) {
        return {std::move(buffer), static_cast<T*>(ptr)};
    }
    throw std::bad_alloc();
}

template<typename T>
void AlignedDataBlob<T>::print(std::ostream& os) const {
    // 检查shape数组的有效性
    if (shape.empty() || std::any_of(shape.begin(), shape.end(), [](size_t s) { return s == 0; })) {
        os << "Invalid shape dimensions." << std::endl;
        return;
    }

    std::function<void(const T*, const std::vector<size_t>&, size_t, const std::string&)> print_recursive;
    print_recursive = [&](const T* data_, const std::vector<size_t>& shape_, const size_t dim, const std::string& indent) {
        os << "[";
        if (dim == shape_.size() - 1) {
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) os << ", ";
                os << data_[i];
            }
            os << "]";
        } else {
            const auto dim_dt = static_cast<std::vector<size_t>::difference_type>(dim);
            const size_t stride = std::accumulate(shape_.begin() + dim_dt + 1, shape_.end(), 1ULL, std::multiplies<>());
            std::string new_indent = indent + " ";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) os << "\n" << new_indent;
                print_recursive(data_ + i * stride, shape_, dim + 1, new_indent);
                if (i < shape_[dim] - 1) os << ",";
            }
            os << "]";
        }
    };
    print_recursive(data, shape, 0, "");
    os << std::endl;
}
    template class AlignedDataBlob<float>;
    template class AlignedDataBlob<double>;
}


