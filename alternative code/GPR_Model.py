import numpy as np
import GPy


def train(X_train, y_train, X_test, n_features):
    n_inputs = n_features-1

    noise = 1
    length = 1
    run_hyperopt_search = True

    kernel = GPy.kern.RBF(input_dim=n_inputs, variance=noise, lengthscale=length)

    gpr = GPy.models.GPRegression(X_train, y_train, kernel)
    if run_hyperopt_search:
        gpr.optimize(messages=False)

    ypred_gp_test_full, cov_test_full = gpr.predict(X_test)

    # randomly sampling from the GPR posterior
    ypred_GPR = np.random.normal(ypred_gp_test_full[0], np.sqrt(cov_test_full[0]))
    for i in range(1, len(X_test)):
        ypred_GPR = np.vstack([ypred_GPR,
                                               np.random.normal(ypred_gp_test_full[i], np.sqrt(cov_test_full[i]))])

    return ypred_GPR
