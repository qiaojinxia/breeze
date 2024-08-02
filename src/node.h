#include <armadillo>
#include <vector>
#include <functional>

namespace Breeze {

    class Node : public std::enable_shared_from_this<Node> {
    public:
        arma::mat value;
        arma::mat grad;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void()> grad_fn;

        explicit Node(const arma::mat& value) : value(value), grad(value.n_rows, value.n_cols, arma::fill::zeros) {}

        // Arithmetic operators for Node objects
        std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator-(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> operator/(const std::shared_ptr<Node>& rhs);
        std::shared_ptr<Node> dot(const std::shared_ptr<Node>& rhs);

        // Unary functions
        std::shared_ptr<Node> sin();
        std::shared_ptr<Node> cos();
        std::shared_ptr<Node> exp();
        std::shared_ptr<Node> log();
        std::shared_ptr<Node> sqrt();

        // Arithmetic operators for Node and constant
        std::shared_ptr<Node> operator+(double rhs);
        std::shared_ptr<Node> operator-(double rhs);
        std::shared_ptr<Node> operator*(double rhs);
        std::shared_ptr<Node> operator/(double rhs);

        static std::shared_ptr<Node> create(const arma::mat& value) {
            return std::make_shared<Node>(value);
        }

        static void backward(const std::shared_ptr<Node>& output_node);
    };

} // namespace Breeze
