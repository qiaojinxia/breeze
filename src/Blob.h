#ifndef Blob_H
#define Blob_H
#include <iostream>
#include <vector>


namespace Breeze {
    template <typename T>
    class Blob {
    public:
        explicit Blob(const std::vector<size_t>& shape);
        explicit Blob(size_t total_size);

        // 禁用拷贝构造和赋值操作符
        Blob(const Blob&) = delete;
        Blob& operator=(const Blob&) = delete;


        Blob(Blob&& other) noexcept;
        Blob& operator=(Blob&& other) noexcept;

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

        friend std::ostream& operator<<(std::ostream& os, const Blob& blob) {
            blob.print(os);
            return os;
        }

        // 调整形状
        void reshape(const std::vector<size_t>& new_shape);
        void setDataWithOwnership(std::shared_ptr<T[]> new_data,const std::vector<size_t>& new_shape);
    private:
        T* data;
        std::shared_ptr<T[]> buffer;
        std::vector<size_t> shape;
        size_t total_size = 0;

        static std::pair<std::shared_ptr<T[]>, T*> allocate_aligned(size_t size, size_t alignment);
        [[nodiscard]] size_t calculateIndex(const std::vector<size_t>& indices) const;
        static_assert(std::is_arithmetic_v<T>, "Blob elements must be of arithmetic type");
    };

}

#endif //Blob_H