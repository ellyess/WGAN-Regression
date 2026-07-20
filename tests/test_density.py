import numpy as np

from wgan_regression import density


def test_y_values_returns_slice():
    X = np.linspace(-1, 1, 201)
    y = X ** 2
    got = density.y_values(X, y, value=0.0, bounds=0.05)
    assert len(got) > 0
    # y = x^2 within |x| <= 0.05 stays near zero
    assert np.all(got <= 0.05 ** 2 + 0.02)


def test_y_values_empty_outside_range():
    X = np.linspace(-1, 1, 50)
    y = X.copy()
    got = density.y_values(X, y, value=10.0, bounds=0.1)
    assert len(got) == 0


def test_y2_values_returns_values_from_y_range():
    # The historical two-input slice lookup is approximate (it matches
    # nearby points via searchsorted rather than exact indices), so this
    # test asserts structural properties only.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, 500)
    X2 = rng.uniform(-1, 1, 500)
    y = X + X2
    got = density.y2_values(X, X2, y, value=0.0, bounds=0.2)
    assert isinstance(got, np.ndarray)
    assert len(got) > 0
    assert got.min() >= y.min() and got.max() <= y.max()
