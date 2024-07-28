//
// Created by caomaobay on 2024/7/17.
//

#ifndef NODE_H
#define NODE_H

#include <functional>
#include <iostream>
#include <vector>
#include <memory>

namespace MyBlob {
    class Node : public std::enable_shared_from_this<Node> {
    public:
        double value;
        double grad;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void()> grad_fn;

        explicit Node(const double value) : value(value), grad(0) {}

        std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator-(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator/(const std::shared_ptr<Node>& rhs);

        std::shared_ptr<Node> sin();
        std::shared_ptr<Node> cos();
        std::shared_ptr<Node> exp();
        std::shared_ptr<Node> log();
        std::shared_ptr<Node> sqrt();

        static void backward(const std::shared_ptr<Node>& output_node);
    };
}

#endif // NODE_H
