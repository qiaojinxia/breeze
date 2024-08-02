//
// Created by caomaobay on 2024/7/23.
//

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <armadillo>

namespace Breeze {
    template<typename T>
    class LossFunction {
    public:
        virtual T forward(const T& predictions, const T& targets) = 0;
        virtual T backward(const T& predictions, const T& targets) = 0;
        virtual ~LossFunction() = default;
    };

    template<typename T>
    class MSELoss final : public LossFunction<T> {
    public:
        T forward(const T& predictions, const T& targets) override {
            // MSE = (1/n) * sum((predictions - targets)^2)
            T diff = predictions - targets;
            return arma::mean(arma::mean(arma::square(diff)));
        }

        T backward(const T& predictions, const T& targets) override {
            // dMSE/dpredictions = 2 * (predictions - targets) / n
            T diff = predictions - targets;
            return 2 * diff / predictions.n_elem;
        }
    };

    template<typename T>
    class CrossEntropyLoss final : public LossFunction<T> {
    public:
        T forward(const T& predictions, const T& targets) override {
            // Cross-entropy loss: -sum(targets * log(predictions)) / n
            T log_preds = arma::log(predictions);
            return -arma::mean(arma::sum(targets % log_preds, 1));
        }

        T backward(const T& predictions, const T& targets) override {
            // dCrossEntropy/dpredictions = -targets / predictions / n
            return -targets / predictions / predictions.n_elem;
        }
    };
}

#endif //LOSS_FUNCTIONS_H
