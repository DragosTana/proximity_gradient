
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression
    from logistic_regression import MyLogisticRegression
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt



    dataset_sizes = [100, 500, 1000, 5000, 10000]
    C = [1, 5, 10]

    errors = []
    my_accuracies = []
    accuracies = []
    for n in C:

        error = []
        acc = []
        my_acc = []
        for i in range(50):
            print(f"Dataset size: {n}, iteration: {i}")

            X, y = make_classification(n, n_features=20, n_informative=20, n_redundant=0, n_repeated=0, n_classes=2)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            penalty = "l1"
            solver = "liblinear"
            lr = LogisticRegression(penalty=penalty,
                                    C=n,
                                    solver=solver,
                                    max_iter=5000,
                                    tol=1e-5,
                                    verbose=0,
                                    fit_intercept=False)

            my_lr = MyLogisticRegression(penalty=penalty,
                                        C=n,
                                        solver="coordinate_descent",
                                        max_iter=5000,
                                        tol=1e-5,
                                        verbose=0,
                                        fit_intercept=False,
                                        reformulated=True)

            my_coef = my_lr.fit(X_train, y_train).coef_
            coef = lr.fit(X_train, y_train).coef_

            relative_error = np.linalg.norm(my_coef - coef) / np.linalg.norm(coef)

            acc.append(accuracy_score(y_test, lr.predict(X_test)))
            my_acc.append(accuracy_score(y_test, my_lr.predict(X_test)))
            error.append(relative_error)

        errors.append(error)
        accuracies.append(np.mean(acc))
        my_accuracies.append(np.mean(my_acc))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(C, accuracies, label="sklearn")
    ax[0].plot(C, my_accuracies, label="my_lr")
    ax[0].set_xlabel("C")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].boxplot(errors)
    ax[1].set_xticklabels(C)
    ax[1].set_xlabel("C")
    ax[1].set_ylabel("Error")
    plt.show()
