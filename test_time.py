
from time import time_ns


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression
    from logistic_regression import MyLogisticRegression
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    dataset_sizes = [100, 500, 1000, 5000, 10000, 100000, 500000]
    lambda_ = [1, 5, 10]
    features = [10, 20, 50, 100, 500]
    trial = 100
    times = np.zeros((len(features), len(dataset_sizes), trial))
    my_times = np.zeros((len(features), len(dataset_sizes), trial))


    for f in features:
        for n in dataset_sizes:

            lr = LogisticRegression(penalty="l1",
                                    C=1,
                                    solver="liblinear",
                                    max_iter=1000,
                                    tol=1e-4,
                                    verbose=0,
                                    fit_intercept=False)

            my_lr = MyLogisticRegression(penalty="l1",
                                        C=1,
                                        solver="coordinate_descent",
                                        max_iter=1000,
                                        tol=1e-4,
                                        verbose=0,
                                        fit_intercept=False,
                                        reformulated=True)
            print(f"Dataset size: {n}, features: {f}")

            for i in tqdm(range(trial)):
                X, y = make_classification(n, n_features=f, n_informative=f, n_redundant=0, n_repeated=0, n_classes=2)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

                start = time_ns()
                lr.fit(X_train, y_train)
                end = time_ns()
                times[features.index(f), dataset_sizes.index(n), i] = end - start

                start = time_ns()
                my_lr.fit(X_train, y_train)
                end = time_ns()
                my_times[features.index(f), dataset_sizes.index(n), i] = end - start

    np.save("times_features_reformulated.npy", times)
    np.save("my_times_features_cd_reformulated.npy", my_times)
    print("Done!")
