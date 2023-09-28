import numpy as np
from scipy.special import expit
from numba import jit, prange

class LossLogisticRegression:
    """
    Class for the logistic regression loss function.
    
    ## Parameters:
    
    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float. 
        Like in support vector machines, smaller values specify stronger regularization.
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization.
    """
    def __init__(
        self, 
        C=1.0, 
        penalty="l2"
        ):
        self.C = C  
        self.penalty = penalty  
        self.coef_ = None  


    def _logistic_loss(self, X, y, coef): 
        z = np.dot(X, coef.T)
        log_loss = np.sum(np.log(1 + np.exp(-y * z)))
        return log_loss

    def _l1_regularization(self, coef):
        l1_loss = np.sum(np.abs(coef))
        return l1_loss
    
    def _l2_regularization(self, coef):
        l2_loss = 0.5 * np.sum(coef**2)
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