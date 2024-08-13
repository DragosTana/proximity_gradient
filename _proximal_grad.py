# Author: Dragos Tanasa
# Description: Implementation of proximal gradient descent solver for Logistic Regression.

import numpy as np
from _loss import LossLogisticRegression
from sklearn.metrics import log_loss
from scipy.special import xlogy, expit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(w, X, y):
    z = (X @ w.T).ravel()
    expit(z, out=z)
    loss = log_loss(y, z, normalize=False)
    return loss, z

def gradient(w, X, y):
    m = X.shape[0]
    z = (X @ w.T).ravel()
    expit(z, out=z)
    error = z - y
    return (1/m) * X.T @ error

def soft_thresholding(v, lambda_):
    return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)

def coordinate_descent_l1_logistic_regression(X,
                                              y,
                                              lambda_,
                                              max_iter=1000,
                                              tol=1e-4,
                                              fit_intercept=True):

    n_samples, n_features = X.shape

    # Initialize weights and intercept
    weights = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        weights_old = weights.copy()
        intercept_old = intercept

        # Update intercept if fit_intercept is True
        if fit_intercept:
            z = np.dot(X, weights) + intercept
            p = sigmoid(z)
            intercept_gradient = np.sum(p - y)
            intercept_hessian = np.sum(p * (1 - p))

            if intercept_hessian != 0:
                intercept -= intercept_gradient / intercept_hessian

        # Update weights
        for j in range(n_features):
            X_j = X[:, j]
            z = np.dot(X, weights) + (intercept if fit_intercept else 0)
            p = sigmoid(z)

            gradient = np.dot(X_j, (p - y)) + lambda_ * np.sign(weights[j])
            hessian = np.dot(X_j**2, p * (1 - p)) + lambda_

            if hessian == 0:
                continue

            delta = gradient / hessian
            weights[j] -= delta

            # Apply soft-thresholding to weights but not to intercept
            weights[j] = soft_thresholding(weights[j], lambda_ / hessian)

        # Check for convergence
        if np.linalg.norm(weights - weights_old) < tol and (not fit_intercept or abs(intercept - intercept_old) < tol):
            break

    if fit_intercept:
        return weights, intercept, max_iter
    else:
        return weights, 0.0, max_iter

def proximal_gradient(X, y, lambda_, max_iter, tol):
    n = X.shape[1]
    w = np.zeros(n)
    for k in range(max_iter):
        v = w - 1 * gradient(w, X, y)
        x = soft_thresholding(v, lambda_ * 1)
        if np.linalg.norm(x - w) < tol:
            return x, None, k

    return x, 0.0, max_iter
