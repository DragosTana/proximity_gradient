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

    def _logistic_loss(self, X, y, coef): 
        z = (X @ coef.T).ravel()
        expit(z, out=z)
        loss = -(xlogy(y, z) + xlogy(1 - y, 1 - z)).sum() 
        if self.normalize:
            self.n = len(y)
            loss /= self.n
        return loss

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

    def _total_loss(self, coef, X, y):
        log_loss = self._logistic_loss(X, y, coef.T)
        reg_loss = 0.0

        if self.penalty == "l1":
            reg_loss = self.C * self._l1_regularization(coef)
        elif self.penalty == "l2":
            reg_loss = 0.5 * self.C * self._l2_regularization(coef)

        total_loss = log_loss + reg_loss
        return total_loss

    def _gradient(self, coef, X, y):
        z = np.dot(X, coef.T)
        exp_z = np.exp(-y * z)
        grad_log_loss = -y * exp_z / (1 + exp_z)

        reg_grad = np.zeros_like(coef)
        if self.penalty == "l1":
            reg_grad = self.C * np.sign(coef)
        elif self.penalty == "l2":
            reg_grad = self.C * coef

        total_grad = np.dot(X.T, grad_log_loss) + reg_grad
        return total_grad
    
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from time import time
    
    
    X, y = make_classification(100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    lr = LogisticRegression(penalty = "l2", C=1.0, solver="saga", max_iter=1000)
    lr.fit(X_train, y_train)
    
    loss = LossLogisticRegression(C=1.0)
    
    y_pred = lr.predict_proba(X_test)
    
    print("sklearn log loss: ", log_loss(y_test, lr.predict_proba(X_test), normalize=False))
    loss = loss._logistic_loss(X_test, y_test, lr.coef_)
    print("LossLogisticRegression log loss: ", loss)
  
    
    
    