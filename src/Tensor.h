#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include "TensorOps.h"

namespace Breeze {
    // 前向声明
    template<typename T>
    class AlignedDataBlob;

    // 设备枚举
    enum class Device {
        CPU,
        GPU
    };

    // 张量类
    template<typename T>
    class Tensor {
    protected:
        std::vector<size_t> shape;
        Device device;
        std::shared_ptr<TensorOps<T>> ops;
    public:
        virtual std::shared_ptr<Tensor> operator*(const Tensor& other) const = 0 ;

        Tensor(const std::vector<size_t>& shape, const Device device)
            : shape(shape), device(device) {
        }

        virtual ~Tensor() = default;

        virtual void resize(std::vector<size_t>) = 0;

        // 纯虚函数
        virtual T* data() = 0;
        virtual const T* data() const = 0;
        [[nodiscard]] virtual size_t size() const = 0;
        virtual void to_cpu() = 0;
        virtual void to_gpu() = 0;
        virtual void print(std::ostream& os) const = 0;

        virtual void fill(T value) const = 0;

        // 共用函数
        [[nodiscard]] const std::vector<size_t>& get_shape() const { return shape; }
        [[nodiscard]] Device get_device() const { return device; }

        // 辅助函数
        [[nodiscard]] size_t num_elements() const {
            size_t total = 1;
            for (const size_t dim : shape) {
                total *= dim;
            }
            return total;
        }

        // 友元函数：重载输出运算符
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
            tensor.print(os);
            return os;
        }
    };
}

#endif //MATRIX_H