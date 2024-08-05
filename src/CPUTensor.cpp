//
// Created by mac on 2024/7/31.
//

#include "CPUTensor.h"
#include <iostream>

namespace Breeze {
    template<typename T>
     std::shared_ptr<Tensor<T>> CPUTensor<T>::matmul(const Tensor<T>& rhs) const {
        return this->ops->matmul(*this, rhs);
    }

    template<typename T>
     std::shared_ptr<Tensor<T>> CPUTensor<T>::operator+(const Tensor<T>& rhs) const {
        return this->ops->add(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator-(const Tensor<T>& rhs) const {
        return this->ops->subtract(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator*(const Tensor<T>& rhs) const {
        return this->ops->multiply(*this, rhs);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> CPUTensor<T>::operator/(const Tensor<T>& rhs) const {
        return this->ops->divide(*this, rhs);
    }

    template<typename T>
    T* CPUTensor<T>::data() {
        return blob.getData();
    }

    template<typename T>
    const T* CPUTensor<T>::data() const {
        return blob.getData();
    }

    template<typename T>
    void CPUTensor<T>::to_cpu() {
        std::cout << "Already on CPU" << std::endl;
    }

    template<typename T>
    void  CPUTensor<T>::to_gpu()  {
        std::cout << "to_gpu not implemented for CPUTensor" << std::endl;
    }


    template<typename T>
    void  CPUTensor<T>::resize(std::vector<size_t> shape) {
        this->shape = shape;
        this->blob.reshape(shape);
    }

    template<typename T>
    void  CPUTensor<T>::print(std::ostream& os) const  {
        blob.print(os);
    }

    template<typename T>
    void  CPUTensor<T>::fill(T value) const  {
        blob.fill(value);
    }

    template<typename T>
    [[nodiscard]] size_t CPUTensor<T>::size() const {
        return blob.getTotalSize();
    }


    template class CPUTensor<float>;
    template class CPUTensor<double>;
}
