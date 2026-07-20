import numpy as np

from wgan_regression import metrics


def _paired_sets(shift=0.0, n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 1))
    y_ref = np.sin(X) + 0.1 * rng.standard_normal(X.shape)
    y_cmp = np.sin(X) + shift + 0.1 * rng.standard_normal(X.shape)
    return X, y_ref, y_cmp


def test_conditional_wasserstein_discriminates():
    X, y_ref, y_same = _paired_sets(shift=0.0)
    _, _, y_shifted = _paired_sets(shift=1.0)
    w_same = metrics.conditional_wasserstein(X, y_ref, X, y_same)
    w_shifted = metrics.conditional_wasserstein(X, y_ref, X, y_shifted)
    assert w_same < 0.2
    assert w_shifted > 0.7


def test_conditional_wasserstein_empty_slices_gives_nan():
    X, y_ref, y_cmp = _paired_sets()
    result = metrics.conditional_wasserstein(
        X, y_ref, X, y_cmp, x_values=[99.0])
    assert np.isnan(result)


def test_mmd_discriminates():
    X, y_ref, y_same = _paired_sets(shift=0.0, n=300)
    _, _, y_shifted = _paired_sets(shift=1.0, n=300)
    joint_ref = np.hstack([X, y_ref])
    m_same = metrics.mmd_rbf(joint_ref, np.hstack([X, y_same]))
    m_shifted = metrics.mmd_rbf(joint_ref, np.hstack([X, y_shifted]))
    assert m_same < 0.01
    assert m_shifted > 0.05


def test_mmd_identical_sets_is_zero():
    a = np.random.default_rng(0).normal(size=(100, 2))
    assert abs(metrics.mmd_rbf(a, a)) < 1e-9


def test_kde_nll_discriminates():
    X, y_ref, y_same = _paired_sets(shift=0.0)
    _, _, y_shifted = _paired_sets(shift=1.0)
    joint_ref = np.hstack([X, y_ref])
    nll_same = metrics.kde_nll(joint_ref, np.hstack([X, y_same]))
    nll_shifted = metrics.kde_nll(joint_ref, np.hstack([X, y_shifted]))
    assert nll_same < nll_shifted
