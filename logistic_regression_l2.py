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
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, X, y, w):
        logits = y * np.dot(X, w)
        loss_terms = np.logaddexp(0, -logits)

        total_loss = 0.0
        for i in range(len(loss_terms)):
            total_loss += loss_terms[i]

        reg = 0.0
        for wi in w:
            reg += wi * wi
        regularization = (self.lambda_reg / 2) * reg

        return total_loss + regularization

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

    def armijo_line_search(self, grad, X, y, direction, alpha_prev, alpha_max=ARMIJO_ALPHA, gamma=ARMIJO_GAMMA, delta=ARMIJO_DELTA):
        delta_0 = min(alpha_max, alpha_prev/delta)
        alpha = delta_0
        current_loss = self.compute_loss(X, y, self.w)
        while True:
            w_new = np.copy(self.w) + alpha * direction
            loss_new = self.compute_loss(X, y, w_new)
            if loss_new <= current_loss + gamma * alpha * np.dot(grad, direction):
                break
            alpha *= delta
        return alpha

    def wolfe_line_search(self, grad, X, y, direction, alpha_max=1, gamma=WOLFE_GAMMA, delta=WOLFE_DELTA, sigma=0.2):
        alpha = alpha_max
        current_loss = self.compute_loss(X, y, self.w)
        while alpha > 1e-12:
            w_new = np.copy(self.w) + alpha * direction
            loss_new = self.compute_loss(X, y, w_new)
            grad_new = self.compute_loss_gradient(X, y, w_new)
            if (loss_new <= current_loss + gamma * alpha * np.dot(grad, direction) and
                    np.abs(np.dot(grad_new, direction)) <= sigma * np.abs(np.dot(grad, direction))):
                break
            alpha *= delta
        return alpha

    def gradient_descent_armijo(self, X, y, tol=1e-4):
        k = 0
        alpha_prev = ARMIJO_ALPHA
        start_time = time.time()
        grad = self.compute_loss_gradient(X, y, self.w)
        while np.linalg.norm(grad) > tol:
            k += 1
            direction = -grad
            lr = self.armijo_line_search(grad, X, y, direction, alpha_prev=alpha_prev)
            alpha_prev = lr
            self.w = np.copy(self.w) + lr * direction
            grad = self.compute_loss_gradient(X, y, self.w)
        return (time.time() - start_time), k

    def conjugate_gradient_wolfe(self, X, y, tol=1e-4):
        k = 0
        start_time = time.time()
        grad = self.compute_loss_gradient(X, y, self.w)
        direction = -grad
        while np.linalg.norm(grad) > tol:
            lr = self.wolfe_line_search(grad, X, y, direction)
            self.w = np.add(self.w, lr * direction)
            grad_new = self.compute_loss_gradient(X, y, self.w)
            beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
            grad = grad_new
            direction = -grad + beta * direction
            k += 1
        return (time.time() - start_time), k