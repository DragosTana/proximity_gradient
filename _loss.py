import numpy as np
from numba import jit, prange
from sklearn.metrics import log_loss
from scipy.special import xlogy, expit
from _utils import fast_matmul

class LossLogisticRegression:
    """
    Class for the logistic regression loss function.

    ## Parameters:

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization.
    normalize : bool, default: False
        If True, the loss is normalized by the number of samples.
    """
    def __init__(
        self,
        C=1.0,
        penalty="l2",
        normalize=False
        ):
        self.C = C
        self.penalty = penalty
        self.normalize = normalize

    def _logistic_loss(self, coef, X, y):
        z = (X @ coef.T).ravel()
        #z = fast_matmul(X, coef).ravel()
        expit(z, out=z)
        loss = -(xlogy(y, z) + xlogy(1 - y, 1 - z)).sum()
        if self.normalize:
            self.n = len(y)
            loss /= self.n
        return loss, z

    def _logistic_loss_sklearn(self, coef, X, y):
        z = (X @ coef.T).ravel()
        #z = fast_matmul(X, coef).ravel()
        expit(z, out=z)
        loss = log_loss(y, z, normalize=self.normalize)
        return loss, z

    def _l1_regularization(self, coef):
        l1_loss = np.sum(np.abs(coef))
        if self.normalize:
            l1_loss /= self.n
        return l1_loss

    def _l2_regularization(self, coef):
        l2_loss = 0.5 * np.sum(coef**2)
        if self.normalize:
            l2_loss /= self.n
        return l2_loss

    def loss(self, coef, X, y):
        if self.penalty == "l1":
            return self._logistic_loss_sklearn(coef, X, y) + self.C * self._l1_regularization(coef)
        elif self.penalty == "l2":
            return self._logistic_loss_sklearn(coef, X, y) + 0.5 * self.C * self._l2_regularization(coef)
        else:
            return self._logistic_loss_sklearn(coef, X, y)

    def _gradient(self, coef, X, y):
        z = (X @ coef.T).ravel()
        #z = fast_matmul(X, coef).ravel()
        expit(z, out=z)
        error = z - y
        grad_log_loss = X.T @ error
        #grad_log_loss = fast_matmul(X, error)
        grad_reg = np.zeros_like(coef)
        if self.penalty == "l1":
            grad_reg = self.C * np.sign(coef)
        elif self.penalty == "l2":
            grad_reg = self.C * coef
        gradient = grad_log_loss + grad_reg
        if self.normalize:
            gradient /= len(y)
        return gradient.ravel()

    def loss_gradient(self, coef, X, y):
        loss, _ = self._logistic_loss(coef, X, y)
        grad = self._gradient(coef, X, y)
        return loss, grad

    def reformulated_loss_gradient(self, uv, X, y):
        u, v = uv[:len(uv)//2], uv[len(uv)//2:]
        input = u - v
        loss, _ = self._logistic_loss_sklearn(input, X, y)
        loss += self.C * np.sum(u + v)
        grad_log_loss = self._gradient(input, X, y)
        grad_u = grad_log_loss + self.C * np.ones_like(u)
        grad_v = -grad_log_loss + self.C * np.ones_like(v)
        return loss, np.concatenate((grad_u, grad_v))
