import numpy as np
import scipy.optimize as opt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR

def compute_loss(data: np.array, labels: np.array, weights: np.array, regularization: str = None) -> float:
    """
    Computes the logistic loss of the data given the weights.
    
    ## Parameters:
    data: np.array
        The data to be used for computing the loss.
    labels: np.array
        The labels of the data.
    weights: np.array
        The weights to be used for computing the loss.
    regularization: str
        The type of regularization to be used. Can be "l1", "l2", or "none".
        
    ## Returns:
    loss: float
        The logistic loss of the data given the weights.
    """
    if regularization == "l1":
        return np.sum(np.log(1 + np.exp(-np.multiply(labels, np.dot(data, weights))))) + np.sum(np.abs(weights))
    elif regularization == "l2":
        return np.sum(np.log(1 + np.exp(-np.multiply(labels, np.dot(data, weights))))) + np.sum(np.square(weights))
    elif regularization == "none":
        return np.sum(np.log(1 + np.exp(-np.multiply(labels, np.dot(data, weights)))))
    else:
        raise ValueError("Invalid regularization parameter.")
    
def compute_loss_gradient():
    pass

def loss(weights: np.array, data: np.array, labels: np.array ) -> float:
    return np.sum(np.log(1 + np.exp(-np.multiply(labels, np.dot(data, weights))))) + np.sum(np.square(weights))
    
class LogisticRegression:
    """
    Logistic regression model.
    """
    
    def __init__(self) -> None:
        self.weights = None
        
    def fit(self, data: np.array, labels: np.array) -> None:
        """
        Fits the logistic regression model to the data.
        
        ## Parameters:
        data: np.array
            The data to be used for fitting the model.
        labels: np.array
            The labels of the data.
        """
        
        # Initialize weights
        self.weights = np.zeros(data.shape[1])
        
        # Optimize weights
        self.weights = opt.minimize(loss, self.weights, args=(data, labels), method="L-BFGS-B", tol=0.0001).x
        
    def predict(self, data: np.array) -> np.array:
        """
        Predicts the labels of the data.
        
        ## Parameters:
        data: np.array
            The data to be used for prediction.
        
        ## Returns:
        predictions: np.array
            The predicted labels.
        """
        return np.sign(np.dot(data, self.weights))
    
class LogisticRegression2(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01, max_iter=10000000, tol=1e-9):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, theta, X, y):
        m = len(y)
        h = self.sigmoid(X @ theta)
        epsilon = 1e-5  # for numerical stability
        loss = (-1 / m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
        return loss

    def gradient(self, theta, X, y):
        m = len(y)
        h = self.sigmoid(X @ theta)
        grad = (1 / m) * X.T @ (h - y)
        return grad

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        # Minimize the loss function using scipy's minimize
        res = opt.minimize(fun=self.loss, x0=self.coef_, args=(X, y), method="L-BFGS-B", jac = self.gradient, tol=self.tol)

        self.coef_ = res.x[1:]  # Exclude the bias term
        self.intercept_ = res.x[0]  # Set the bias term

        return self

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return (self.sigmoid(X @ np.concatenate(([self.intercept_], self.coef_))) >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return self.sigmoid(X @ np.concatenate(([self.intercept_], self.coef_)))
    
def main():
    X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logistic_regression = LR(penalty = 'l2', solver = 'lbfgs', max_iter = 10000000)
    logistic_regression.fit(X, y)
    
    my_logistic_regression = LogisticRegression2()
    my_logistic_regression.fit(X, y)

    print("Sklearn LOSS: ", loss(x_train, y_train, logistic_regression.coef_))
    print("My LOSS: ", loss(x_train, y_train, my_logistic_regression.coef_))
    print("Weights: ", logistic_regression.coef_)
    print("My weights: ", my_logistic_regression.coef_)
    print("Sklearn accuracy: ", logistic_regression.score(x_test, y_test))
    print("My accuracy: ", np.mean(my_logistic_regression.predict(x_test) == y_test))
    print(" ")
    
if __name__ == "__main__":
    #set seed
    for i in range(5):
        main()