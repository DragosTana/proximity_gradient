
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from logistic_regression import MyLogisticRegression
    from time import time
    import numpy as np

    X, y = make_classification(10000, n_features=60, n_informative=60, n_redundant=0, n_repeated=0, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    penalty = "l1"
    solver = "liblinear"
    lr = LogisticRegression(penalty=penalty,
                            C=1.0,
                            solver=solver,
                            max_iter=1000,
                            tol=1e-4,
                            verbose=0,
                            fit_intercept=True)

    my_lr = MyLogisticRegression(penalty=penalty,
                                C=1.0,
                                solver="proximal_grad",
                                max_iter=1000,
                                tol=1e-4,
                                verbose=0,
                                fit_intercept=True,
                                reformulated=False)

    start = time()
    lr.fit(X_train, y_train)
    print("Sklearn time: ", time() - start)

    #y_train[y_train == 0] = -1
    start = time()
    my_lr.fit(X_train, y_train)
    print("My time: ", time() - start)

    print("Sklearn score: ", lr.score(X_test, y_test))
    print("My score: ", my_lr.score(X_test, y_test))

    my_proba = my_lr.predict_proba(X_test)
    proba = lr.predict_proba(X_test)

    print("Sklearn intercept: ", lr.intercept_)
    print("My inctercep: ", my_lr.intercept_)

    if np.allclose(lr.coef_, my_lr.coef_, atol=1e-3):
        print("my_coef and coef are equal")
    else:
        print("my_coef and coef are not equal")

    if np.allclose(my_proba, proba, atol=1e-4):
        print("my_proba and proba are equal")
    else:
        print("my_proba and proba are not equal")
