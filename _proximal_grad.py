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

def coordinate_descent_l1_logistic_regression(X, y, lambda_, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        weights_old = weights.copy()

        for j in range(n_features):
            X_j = X[:, j]
            z = np.dot(X, weights)
            p = sigmoid(z)

            gradient = np.dot(X_j, (p - y)) + lambda_ * np.sign(weights[j])
            hessian = np.dot(X_j**2, p * (1 - p)) + lambda_

            if hessian == 0:
                continue

            delta = gradient / hessian
            weights[j] -= delta

            weights[j] = soft_thresholding(weights[j], lambda_ / hessian)

        if np.linalg.norm(weights - weights_old) < tol:
            break

    return weights, None, max_iter

def proximal_gradient(X, y, lambda_, max_iter, tol):
    n = X.shape[1]
    w = np.zeros(n)
    for k in range(max_iter):
        v = w - 1 * gradient(w, X, y)
        x = soft_thresholding(v, lambda_ * 1)
        if np.linalg.norm(x - w) < tol:
            return x, None, k

    return x, None, max_iter
