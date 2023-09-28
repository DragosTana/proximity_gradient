# Author: Dragos Tanasa
# Description: Implementation of proximal gradient descent solver for Logistic Regression.

import numpy as np

def proximity_operator(coef, C):
    new_coef = np.zeros_like(coef)
    new_coef[1:] = np.sign(coef[1:]) * np.maximum(np.abs(coef[1:]) - C, 0)
    
    return new_coef

def proximal_gradient(X, y, C, penalty, fit_intercept, max_iter, tol):

    coef = np.zeros(X.shape[1])
    
    for i in range(max_iter):
        coef = proximity_operator