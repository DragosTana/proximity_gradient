# Author: Dragos Tanasa
# Description: Implementation of Logistic Regression classifier compatible with sklearn.

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import scipy.optimize as opt
from scipy.special import xlogy, expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.svm._base import _fit_liblinear


from _loss import LossLogisticRegression
from proximal_grad import proximal_gradient
import _utils as u

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier.
    
    ## Parameters:
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. 
        
    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float. 
        Like in support vector machines, smaller values specify stronger regularization.
        
    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        
    solver : {'lbfgs', 'liblinear', 'proximal_gra'}, default: 'lbfgs'
        Algorithm to use in the optimization problem.
        
        ..warning::
            The choice of the solver depends on the penalty chosen.
            Supported penalties by solver are:
            
                - 'lbfgs'           -   'l2', None
                - 'liblinear'       -   'l1', 'l2'
                - 'proximal_grad'   -   'l1'
                
    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
        
    maxc_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.
        
    tol : float, default: 1e-4
        Tolerance for stopping criteria.
        
    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
    """
    
    def __init__(
        self,
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        solver='lbfgs',
        dual=False,
        max_iter=100,
        tol=1e-4,
        verbose=0,
    ):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.dual = dual
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.coef_ = None
        
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        
        ## Parameters:
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
            
        y : array-like of shape (n_samples,)
            Target vector relative to X.
            
        ## Returns:
        self : object
            Returns self, fitted estimator.
        """
        
        solver = u._check_solver(self.solver, self.penalty, self.dual)
        X, y = check_X_y(X, y, accept_sparse='csr', order="C", dtype=np.float64)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )
            
        loss = LossLogisticRegression(C=self.C, penalty=self.penalty)
        fun = loss._total_loss
        grad = loss._gradient
            
        if solver == 'lbfgs':
            optimizer = opt.minimize(
                fun=fun,
                x0=np.zeros(X.shape[1]),
                args=(X, y),
                method='L-BFGS-B',
                jac=grad,
                tol=self.tol,
                options={
                    'maxiter': self.max_iter,
                    'disp': self.verbose
                }
            )
            self.coef_ = optimizer.x
            self.n_iter_ = optimizer.nit
            
        elif solver == 'liblinear':
            self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
                X = X,
                y = y,
                C = self.C,
                fit_intercept = self.fit_intercept,
                intercept_scaling = 1,
                class_weight = None,
                penalty = self.penalty,
                dual = False,
                verbose = self.verbose,
                max_iter = self.max_iter,
                tol = self.tol,
                random_state = None,
                multi_class = 'ovr',
                loss = 'logistic_regression',
                epsilon = 1e-4,
                sample_weight = None,
            )
            
        elif solver == 'proximal_grad':
            self.coef_, self.intercept_, self.n_iter_ = proximal_gradient(
                X = X,
                y = y,
                C = self.C,
                fit_intercept = self.fit_intercept,
                penalty = self.penalty,
                max_iter = self.max_iter,
                tol = self.tol,
            )
            
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        ## Parameters:
        X : array-like of shape (n_samples, n_features)
            Samples.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', order="C")
        y_pred = np.dot(X, self.coef_.T)
        y_pred = np.sign(y_pred)
        y_pred[y_pred == -1] = 0
        return y_pred
            
    def predict_proba(self, X):
        """
        Probability estimates.
        
        The returned estimates for all classes are ordered by the label of classes.
        
        ## Parameters:
        X : array-like of shape (n_samples, n_features)
            Samples.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', order="C")
        y_pred = (X @ self.coef_.T).ravel()
        expit(y_pred, out=y_pred)
        return np.vstack((1 - y_pred, y_pred)).T
            
            
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from time import time
    
    X, y = make_classification(10000, n_features=60, n_informative=60, n_redundant=0, n_repeated=0, n_classes=2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    lr = LogisticRegression(penalty="l2", C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4, verbose=0, fit_intercept=False, )
    my_lr = MyLogisticRegression(penalty="l2", C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4, verbose=0, fit_intercept=False)
    
    start = time()
    lr.fit(X_train, y_train)
    print("Sklearn time: ", time() - start)
    
    #y_train[y_train == 0] = -1
    start = time()
    my_lr.fit(X_train, y_train)
    print("My time: ", time() - start)
    
    my_y_pred = my_lr.predict(X_test)
    y_pred = lr.predict(X_test)
    
    print("My accuracy: ", accuracy_score(y_test, my_y_pred))
    print("Sklearn accuracy: ", accuracy_score(y_test, y_pred))
    
    my_proba = my_lr.predict_proba(X_test)
    proba = lr.predict_proba(X_test)
    
    if np.allclose(my_proba, proba, atol=1e-4):
        print("my_proba and proba are equal")
    else:
        print("my_proba and proba are not equal")

