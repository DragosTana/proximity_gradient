# Author: Dragos Tanasa
# Description: Implementation of proximal gradient descent solver for Logistic Regression.

import numpy as np
from _loss import LossLogisticRegression

def proximal_operator(w, C, penalty):
    if penalty == 'l2':
        return w / (1 + 2 * C)
    elif penalty == 'l1':
        return np.sign(w) * np.maximum(np.abs(w) - C, 0)

def proximal_gradient(X, y, C, fit_intercept, penalty, max_iter, tol):

    n_samples, n_features = X.shape
    coef = np.zeros(n_features + 1) if fit_intercept else np.zeros(n_features)
    intercept = 0.0
    n_iter = 0
    
    loss = LossLogisticRegression(C=C, penalty=penalty)
    fun = loss._total_loss
    grad = loss._gradient
    
    while n_iter < max_iter:
        
        grad_coef = grad(X, y, coef)
        coef_new = coef - grad_coef
        coef_new = proximal_operator(coef_new, C, penalty)
        
        coef_diff = np.linalg.norm(coef_new - coef)
        if coef_diff < tol:
            break
        
        coef = coef_new
        n_iter += 1
        
    return coef, intercept, n_iter
        