#ifndef CPUTENSOR_H
#define CPUTENSOR_H

#include "Tensor.h"
#include "Blob.h"
#include <iostream>
#include "CPUTensorOps.h"

namespace Breeze {
    template<typename T>
    class CPUTensor final : public Tensor<T> {
    public:
        explicit CPUTensor(const std::vector<size_t>& shape)
            : Tensor<T>(shape, Device::CPU), blob(std::make_shared<Blob<T>>(shape)){
            this->ops = std::make_shared<CPUTensorOps<T>>();
        }

        T* data() override;

        void setData(std::shared_ptr<Blob<T>> new_blob) override {
            this->shape = new_blob->getShape();
            blob = std::move(new_blob);
        }

        const T* data() const override;

        [[nodiscard]] size_t size() const override;

        void to_cpu() override;

        void to_gpu() override;

        void resize(std::vector<size_t> shape) override;

        void print(std::ostream& os) const override;

        void fill(T value) const override;

        std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) const override;
        std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) const override;

        std::shared_ptr<Tensor<T>> matmul(const Tensor<T>& rhs) const override;

        void broadcast(Tensor<T>& rhs)  override;

    protected:
       std::shared_ptr<Blob<T>> blob;
    };

}

#endif //CPUTENSOR_H