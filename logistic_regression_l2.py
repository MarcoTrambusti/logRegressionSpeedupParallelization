import numpy as np
import time

ARMIJO_ALPHA = 1
ARMIJO_DELTA = 0.5
ARMIJO_GAMMA = 1e-4
WOLFE_DELTA = 0.8
WOLFE_GAMMA = 1e-4
WOLFE_SIGMA = 0.2

# classe logisticreression L2 con metodo di calcolo della loss e gradiente della loss
class LogisticRegressionL2:
    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg
        self.w = None
        self.gradient_time = 0.0
        self.loss_time = 0.0
        self.total_time = 0.0

    @staticmethod
    def sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, X, y, w):
        start = time.time()
        logits = y * np.dot(X, w)
        loss_terms = np.logaddexp(0, -logits)

        total_loss = 0.0
        for i in range(len(loss_terms)):
            total_loss += loss_terms[i]

        reg = 0.0
        for wi in w:
            reg += wi * wi
        regularization = (self.lambda_reg / 2) * reg
        loss = total_loss + regularization

        elapsed = time.time() - start
        self.loss_time += elapsed
        self.total_time += elapsed
        return loss

    def initialize_params(self, n_features):
        self.w = np.zeros(n_features)
        self.gradient_time = 0.0
        self.loss_time = 0.0
        self.total_time = 0.0

    def compute_loss_gradient(self, X, y, w):
        start = time.time()

        logits = X.dot(w)
        z = -y * logits
        sig = self.sigmoid(z)
        r = -y * sig
        grad = X.T.dot(r) + self.lambda_reg * w

        elapsed = time.time() - start
        self.gradient_time += elapsed
        self.total_time += elapsed
        return grad

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

    def wolfe_line_search(self, grad, X, y, direction, alpha_max=1, gamma=WOLFE_GAMMA, delta=WOLFE_DELTA, sigma=WOLFE_SIGMA):
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
        grad = self.compute_loss_gradient(X, y, self.w)
        while np.linalg.norm(grad) > tol:
            k += 1
            direction = -grad
            lr = self.armijo_line_search(grad, X, y, direction, alpha_prev=alpha_prev)
            alpha_prev = lr
            self.w = np.copy(self.w) + lr * direction
            grad = self.compute_loss_gradient(X, y, self.w)
        return k

    def conjugate_gradient_wolfe(self, X, y, tol=1e-4):
        k = 0
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
        return k