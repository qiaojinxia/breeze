#include "node.h"
#include <unordered_set>
#include <armadillo>

using namespace MyBlob;

// Overload for addition with a constant on the right-hand side
std::shared_ptr<Node> Node::operator+(const double rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value + rhs);
    node->parents = {lhs};
    node->grad_fn = [node, lhs]() {
        lhs->grad += node->grad;
    };
    return node;
}

// Overload for addition with a constant on the left-hand side
std::shared_ptr<Node> operator+(const double lhs, const std::shared_ptr<Node>& rhs) {
    return *rhs + lhs;
}

// Overload for subtraction with a constant on the right-hand side
std::shared_ptr<Node> Node::operator-(const double rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value - rhs);
    node->parents = {lhs};
    node->grad_fn = [node, lhs]() {
        lhs->grad += node->grad;
    };
    return node;
}

// Overload for subtraction with a constant on the left-hand side
std::shared_ptr<Node> operator-(const double lhs, const std::shared_ptr<Node>& rhs) {
    auto node = std::make_shared<Node>(lhs - rhs->value);
    node->parents = {rhs};
    node->grad_fn = [node, rhs]() {
        rhs->grad -= node->grad;
    };
    return node;
}

// Overload for multiplication with a constant on the right-hand side
std::shared_ptr<Node> Node::operator*(const double rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value * rhs);
    node->parents = {lhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += rhs * node->grad;
    };
    return node;
}

// Overload for multiplication with a constant on the left-hand side
std::shared_ptr<Node> operator*(const double lhs, const std::shared_ptr<Node>& rhs) {
    return *rhs * lhs;
}

// Overload for division with a constant on the right-hand side
std::shared_ptr<Node> Node::operator/(const double rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value / rhs);
    node->parents = {lhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += (1 / rhs) * node->grad;
    };
    return node;
}

// Overload for division with a constant on the left-hand side
std::shared_ptr<Node> operator/(const double lhs, const std::shared_ptr<Node>& rhs) {
    auto node = std::make_shared<Node>(lhs / rhs->value);
    node->parents = {rhs};
    node->grad_fn = [node, rhs, lhs]() {
        rhs->grad -= (lhs / arma::square(rhs->value)) % node->grad;
    };
    return node;
}

// Existing operator overloads and other member functions...
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
    auto node = std::make_shared<Node>(this->value % rhs->value); // Element-wise multiplication
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += rhs->value % node->grad;
        rhs->grad += lhs->value % node->grad;
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

std::shared_ptr<Node> Node::operator/(const std::shared_ptr<Node>& rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value / rhs->value); // Element-wise division
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += (1 / rhs->value) % node->grad; // Element-wise division
        rhs->grad -= (lhs->value / arma::square(rhs->value)) % node->grad; // Element-wise division and square
    };
    return node;
}

std::shared_ptr<Node> Node::dot(const std::shared_ptr<Node>& rhs) {
    auto lhs = shared_from_this();
    auto node = std::make_shared<Node>(this->value * rhs->value); // Matrix multiplication
    node->parents = {lhs, rhs};
    node->grad_fn = [node, lhs, rhs]() {
        lhs->grad += node->grad * rhs->value.t(); // Gradient with respect to A
        rhs->grad += lhs->value.t() * node->grad; // Gradient with respect to B
    };
    return node;
}

std::shared_ptr<Node> Node::sin() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(arma::sin(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += arma::cos(self->value) % node->grad; // Element-wise multiplication
    };
    return node;
}

std::shared_ptr<Node> Node::cos() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(arma::cos(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad -= arma::sin(self->value) % node->grad; // Element-wise multiplication
    };
    return node;
}

std::shared_ptr<Node> Node::exp() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(arma::exp(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += node->value % node->grad; // Element-wise multiplication
    };
    return node;
}

std::shared_ptr<Node> Node::log() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(arma::log(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += (1 / self->value) % node->grad; // Element-wise division
    };
    return node;
}

std::shared_ptr<Node> Node::sqrt() {
    auto self = shared_from_this();
    auto node = std::make_shared<Node>(arma::sqrt(this->value));
    node->parents = {self};
    node->grad_fn = [node, self]() {
        self->grad += (0.5 / node->value) % node->grad; // Element-wise division
    };
    return node;
}

void Node::backward(const std::shared_ptr<Node>& output_node) {
    output_node->grad.ones(output_node->value.n_rows, output_node->value.n_cols); // Set gradient to ones matrix
    std::vector<std::shared_ptr<Node>> nodes = {output_node};
    std::unordered_set<std::shared_ptr<Node>> visited; // Used to track visited nodes

    while (!nodes.empty()) {
        const auto node = nodes.back();
        nodes.pop_back();
        // If the node has already been processed, skip it
        if (visited.find(node) != visited.end()) {
            continue;
        }
        visited.insert(node); // Mark the current node as visited

        if (node->grad_fn) {
            node->grad_fn();
            for (auto const& parent : node->parents) {
                nodes.push_back(parent);
            }
        }
    }
}
