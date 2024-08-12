import numpy as np
from numba import jit, prange
from sklearn.metrics import log_loss
from scipy.special import xlogy, expit

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
        self.coef_ = None
        self.normalize = normalize

    def _logistic_loss(self, coef, X, y):
        z = (X @ coef.T).ravel()
        expit(z, out=z)
        loss = -(xlogy(y, z) + xlogy(1 - y, 1 - z)).sum()
        if self.normalize:
            self.n = len(y)
            loss /= self.n
        return loss, z

    def _logistic_loss_sklearn(self, coef, X, y):
        z = (X @ coef.T).ravel()
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
        expit(z, out=z)
        error = z - y
        grad_log_loss = X.T @ error
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


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from time import time

    X, y = make_classification(100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LogisticRegression(penalty = "l2", C=1.0, solver="saga", max_iter=1000, fit_intercept=False)
    lr.fit(X_train, y_train)
    print(lr.coef_)
    print(lr.intercept_)
    loss = LossLogisticRegression(C=1.0, penalty=None)

    y_pred = lr.predict_proba(X_test)
    print("sklearn log loss: ", log_loss(y_test, y_pred, normalize=False))
    loss, _ = loss._logistic_loss(lr.coef_, X_test, y_test)
    print("LossLogisticRegression log loss: ", loss)

    loss = LossLogisticRegression(C=1.0)
    grad = loss._gradient(lr.coef_, X_test, y_test)
    print(grad.shape)
    print("LossLogisticRegression gradient: ", grad)
