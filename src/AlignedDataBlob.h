#ifndef AlignedDataBlob_H
#define AlignedDataBlob_H
#include <iostream>
#include <vector>


namespace Breeze {
    template <typename T>
    class AlignedDataBlob {
    public:
        explicit AlignedDataBlob(const std::vector<size_t>& shape);

        // 禁用拷贝构造和赋值操作符
        AlignedDataBlob(const AlignedDataBlob&) = delete;
        AlignedDataBlob& operator=(const AlignedDataBlob&) = delete;

        AlignedDataBlob(AlignedDataBlob&& other) noexcept;
        AlignedDataBlob& operator=(AlignedDataBlob&& other) noexcept;

        T* getData() { return data; }
        const T* getData() const { return data; }

        [[nodiscard]] const std::vector<size_t>& getShape() const { return shape; }
        [[nodiscard]] size_t getDimension() const { return shape.size(); }
        [[nodiscard]] size_t getTotalSize() const { return total_size; }

        // 元素访问
        T& at(const std::vector<size_t>& indices);
        const T& at(const std::vector<size_t>& indices) const;

        // 填充随机数
        void fillRandom(T minValue, T maxValue);

        // 填充数据
        void fill(T value) const{
            std::fill(data, data + total_size, value);
        }

        void print(std::ostream& os = std::cout) const;

        friend std::ostream& operator<<(std::ostream& os, const AlignedDataBlob& blob) {
            blob.print(os);
            return os;
        }

        // 调整形状
        void reshape(const std::vector<size_t>& new_shape);

    private:
        T* data;
        std::unique_ptr<T[]> buffer;
        std::vector<size_t> shape;
        size_t total_size;

        static std::pair<std::unique_ptr<T[]>, T*> allocate_aligned(size_t size, size_t alignment);
        [[nodiscard]] size_t calculateIndex(const std::vector<size_t>& indices) const;
        static_assert(std::is_arithmetic_v<T>, "AlignedDataBlob elements must be of arithmetic type");
    };
}

#endif //AlignedDataBlob_H