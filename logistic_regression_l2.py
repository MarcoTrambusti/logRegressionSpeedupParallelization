import numpy as np
import time

ARMIJO_DELTA = 0.5
WOLFE_DELTA = 0.8
ARMIJO_ALPHA = 1
ARMIJO_GAMMA = 1e-4
WOLFE_GAMMA = 1e-4


# classe logisticreression L2 con metodo di calcolo della loss e gradiente della loss
class LogisticRegressionL2:
    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg
        self.w = None

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y, w):
        logits = y * X.dot(w)
        loss_terms = np.logaddexp(0, -logits)

        return np.sum(loss_terms) + (self.lambda_reg / 2) * np.sum(np.square(w))

    def feed_forward(self, X):
        z = X.dot(self.w)
        A = self.sigmoid(z)
        return A

    def initialize_params(self, n_features):
        self.w = np.zeros(n_features)

    def compute_loss_gradient(self, X, y, w):
        logits = X.dot(w)
        z = -y * logits
        sig = self.sigmoid(z)
        r = -y * sig
        grad = X.T.dot(r)
        return grad + self.lambda_reg * w

    def armijo_line_search(self, grad, X, y, direction, alpha_max=ARMIJO_ALPHA, gamma=ARMIJO_GAMMA, delta=ARMIJO_DELTA):
        alpha = alpha_max
        current_loss = self.compute_loss(X, y, self.w)
        while True:
            w_new = self.w + alpha * direction
            loss_new = self.compute_loss(X, y, w_new)
            if loss_new <= current_loss + gamma * alpha * grad.dot(direction):
                break
            alpha *= delta
        return alpha

    def wolfe_line_search(self, grad, X, y, direction, alpha_max=1, gamma=WOLFE_GAMMA, delta=WOLFE_DELTA, sigma=0.2):
        alpha = alpha_max
        current_loss = self.compute_loss(X, y, self.w)
        while True:
            w_new = self.w + alpha * direction
            loss_new = self.compute_loss(X, y, w_new)
            grad_new = self.compute_loss_gradient(X, y, w_new)
            if loss_new <= current_loss + gamma * alpha * grad.dot(direction) and np.abs(
                    grad_new.dot(direction)) <= sigma * np.abs(grad.dot(direction)):
                break
            alpha *= delta
            if alpha < 1e-12:
                return alpha
        return alpha

    def gradient_descent_armijo(self, X, y, tol=1e-4):
        k = 0
        start_time = time.time()
        grad = self.compute_loss_gradient(X, y, self.w)
        while np.linalg.norm(grad) > tol:
            k += 1
            direction = -grad
            lr = self.armijo_line_search(grad, X, y, direction)
            self.w += lr * direction
            grad = self.compute_loss_gradient(X, y, self.w)
        print(f"Iter {k}, grad_norm: {np.linalg.norm(grad)}")
        return (time.time() - start_time), k

    def conjugate_gradient_wolfe(self, X, y, tol=1e-4):
        k = 0
        start_time = time.time()
        grad = self.compute_loss_gradient(X, y, self.w)
        direction = -grad

        while np.linalg.norm(grad) > tol:
            lr = self.wolfe_line_search(grad, X, y, direction)
            self.w += lr * direction
            grad_new = self.compute_loss_gradient(X, y, self.w)
            beta = np.linalg.norm(grad_new) ** 2 / np.linalg.norm(grad) ** 2
            grad = grad_new
            direction = -grad + beta * direction
            k += 1
        return (time.time() - start_time), k
