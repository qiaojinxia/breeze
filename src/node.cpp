//
// Created by caomaobay on 2024/7/17.
//

#include "node.h"

#include <unordered_set>
using namespace MyBlob;

std::shared_ptr<Node> Node::operator+(const std::shared_ptr<Node>& rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value + rhs->value);
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += node->grad;
        rhs->grad += node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::operator*(const std::shared_ptr<Node>& rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value * rhs->value);
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += rhs->value * node->grad;
        rhs->grad += lhs->value * node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::operator-(const std::shared_ptr<Node>& rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value - rhs->value);
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += node->grad;
        rhs->grad -= node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::operator/(const std::shared_ptr<Node>&  rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value / rhs->value);
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += (1 / rhs->value) * node->grad;
        rhs->grad -= (lhs->value / (rhs->value * rhs->value)) * node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::sin() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(std::sin(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += std::cos(self->value) * node->grad;
    };
    return node;
}


std::shared_ptr<Node> Node::cos() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(std::cos(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad -= std::sin(self->value) * node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::exp() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(std::exp(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += node->value * node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::log() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(std::log(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += (1 / self->value) * node->grad;
    };
    return node;
}

std::shared_ptr<Node> Node::sqrt() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(std::sqrt(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += (0.5 / node->value) * node->grad;
    };
    return node;
}

void Node::backward(const std::shared_ptr<Node>& output_node) {
    output_node->grad = 1.0;
    std::vector<std::shared_ptr<Node>> nodes = {output_node};
    std::unordered_set<std::shared_ptr<Node>> visited; // 用于记录已访问的节点

    while (!nodes.empty()) {
        const auto node = nodes.back();
        nodes.pop_back();
        // 如果节点已经被处理过，则跳过
        if (visited.find(node) != visited.end()) {
            continue;
        }
        visited.insert(node); // 标记当前节点为已访问

        if (node->grad_fn) {
            node->grad_fn();
            for (auto const& parent : node->parents) {
                nodes.push_back(parent);
            }
        }
    }
}
