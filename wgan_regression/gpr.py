"""Gaussian Process Regression baseline (GPy).

GPR is the natural yardstick for the WGAN approach: it is the standard
probabilistic regression model, and like the WGAN it produces a predictive
*distribution* rather than a point estimate. The comparison in the paper
shows where GPR's Gaussian assumptions break down (multi-modal or
multi-valued responses) while the GAN keeps up.

Requires the optional ``GPy`` dependency.
"""

import numpy as np
import GPy


def train(X_train, y_train, X_test, n_features):
    """Fit a GPR with an RBF kernel and sample from its posterior.

    Kernel hyperparameters (variance and lengthscale) are optimised by
    maximum likelihood from a unit initialisation. To compare like with
    like against the GAN, which produces samples rather than means, the returned
    predictions are random draws from the GPR posterior at each test point,
    not the posterior mean.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_inputs)
    y_train : ndarray of shape (n_samples, 1)
    X_test : ndarray of shape (n_test, n_inputs)
    n_features : int
        Total feature count (inputs + outputs); inputs = n_features - 1.

    Returns
    -------
    ndarray of shape (n_test, 1)
        One posterior sample of y per test point.
    """
    n_inputs = n_features - 1

    kernel = GPy.kern.RBF(input_dim=n_inputs, variance=1.0, lengthscale=1.0)

    gpr = GPy.models.GPRegression(X_train, y_train, kernel)
    gpr.optimize(messages=False)

    y_mean, y_cov = gpr.predict(X_test)

    # Draw one sample per test point from N(mean, std).
    ypred_GPR = np.random.normal(y_mean[0], np.sqrt(y_cov[0]))
    for i in range(1, len(X_test)):
        ypred_GPR = np.vstack(
            [ypred_GPR, np.random.normal(y_mean[i], np.sqrt(y_cov[i]))])

    return ypred_GPR
