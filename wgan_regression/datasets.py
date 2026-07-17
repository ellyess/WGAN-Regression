"""Benchmark regression datasets.

Each generator returns ``(X, y)`` as column arrays: ``X`` has shape
``(n, n_inputs)`` and ``y`` has shape ``(n, 1)``. The datasets are chosen to
stress different failure modes of standard regression models:

==========  ========  ===========================================================
Scenario    Inputs    What it tests
==========  ========  ===========================================================
``sinus``   1         Baseline: smooth curve with homoscedastic noise.
``circle``  1         Multi-valued response: two y branches for most x.
``multi``   1         Multi-modal response: two overlapping regimes.
``moons``   1         Two interleaved crescents (scikit-learn's make_moons).
``heter``   1         Heteroscedastic noise growing with x.
``eye``     1         Real data (electrical impedance of an eye, from file).
``3d``      2         Cone surface z = sqrt(x1^2 + x2^2).
``helix``   2         3-D helix: a curve, not a function, in input space.
==========  ========  ===========================================================

Use :func:`get_dataset` to draw independent train/test/validation splits for
any scenario by name.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets

# Repository data folder (works regardless of the caller's working directory).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def gen_sinusoidal(n_instance):
    """Noisy sine wave: y = sin(x) + eps, x in [-4, 4]."""
    noise = 0.2

    X = np.linspace(start=-4, stop=4, num=n_instance).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(*X.shape)

    return X, y


def gen_circle(n_instance):
    """Annulus of points: for most x there are two valid y values.

    Points are sampled at uniform random angles with radius drawn from
    [0.6, 1.0], giving a ring with some thickness.
    """
    t = np.random.random(size=n_instance) * 2 * np.pi - np.pi
    x_ = np.cos(t)
    y_ = np.sin(t)

    for i in range(n_instance):
        length = 1 - np.random.random() * 0.4
        x_[i] = x_[i] * length
        y_[i] = y_[i] * length

    return x_.reshape(-1, 1), y_.reshape(-1, 1)


def get_multimodal(n_instance):
    """Multi-modal dataset: two different y regimes over the same x range.

    Half the points follow a piecewise-linear trend, the other half follow
    sin(10x) + 0.6, so the conditional p(y | x) has two modes.
    """
    x = np.random.rand(int(n_instance / 2), 1)
    y1 = np.ones((int(n_instance / 2), 1))
    y1[x < 0.4] = 1.2 * x[x < 0.4] + 0.2 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y1[np.logical_and(x >= 0.4, x < 0.6)] = (
        0.5 * x[np.logical_and(x >= 0.4, x < 0.6)]
        + 0.01 * np.random.randn(np.sum(np.logical_and(x >= 0.4, x < 0.6))))
    y1[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y2 = np.sin(10 * x) + 0.6 + 0.1 * np.random.randn(*x.shape)

    y = np.array(np.vstack([y1, y2])[:, 0]).reshape((n_instance, 1))
    x = np.tile(x, (2, 1))
    x = np.array(x[:, 0]).reshape((n_instance, 1))

    return x, y


def gen_3d(n_instance):
    """Cone surface: two inputs (x1, x2), output y = sqrt(x1^2 + x2^2)."""
    noise = 0.2
    x = np.linspace(start=-4, stop=4, num=n_instance // 10).reshape(-1, 1)
    x2 = np.linspace(start=-4, stop=4, num=n_instance // 10).reshape(-1, 1)
    x, x2 = (np.meshgrid(x, x2)
             + noise * np.random.randn(*x.shape)
             + noise * np.random.randn(*x2.shape))
    y = np.sqrt(x ** 2 + x2 ** 2)

    x = x.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return np.concatenate((x, x2), axis=1), y


def gen_moons(n_instance):
    """Two interleaved half-circles (scikit-learn's make_moons)."""
    noise = 0.05

    points, _ = sk_datasets.make_moons(n_samples=n_instance, noise=noise)

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)

    return X, y


def gen_helix(n_instance):
    """Helix in 3-D: inputs (cos t, sin t + eps), output t."""
    noise = 0.2
    t = np.linspace(0, 20, n_instance)
    x = np.cos(t)
    x2 = np.sin(t) + noise * np.random.randn(*x.shape)
    y = t

    x = x.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return np.concatenate((x, x2), axis=1), y


def gen_eye(n_instance):
    """Real dataset loaded from ``data/eyedata.csv`` (n_instance ignored)."""
    eye = np.asarray(pd.read_csv(DATA_DIR / "eyedata.csv"))

    X = eye[:, 0].reshape(-1, 1)
    y = eye[:, 1].reshape(-1, 1)

    return X, y


def gen_heteroscedastic(n_instance):
    """Curve whose noise amplitude grows with |x| (heteroscedastic)."""
    theta = np.linspace(0, 2, n_instance)
    X = np.exp(theta) * np.tan(0.1 * theta)
    b = (0.001 + 0.5 * np.abs(X)) * np.random.normal(1, 1, n_instance)
    y = np.exp(theta) * np.sin(0.1 * theta) + b

    return X.reshape(-1, 1), y.reshape(-1, 1)


_SCENARIOS = {
    "sinus": gen_sinusoidal,
    "circle": gen_circle,
    "multi": get_multimodal,
    "3d": gen_3d,
    "moons": gen_moons,
    "helix": gen_helix,
    "eye": gen_eye,
    "heter": gen_heteroscedastic,
}


def get_dataset(n_instance=1000, scenario="sinus", seed=None):
    """Draw independent train, test and validation sets for a scenario.

    Parameters
    ----------
    n_instance : int, optional
        Number of samples per split (ignored by the file-based ``eye``
        scenario).
    scenario : str, optional
        One of ``"sinus"``, ``"circle"``, ``"multi"``, ``"3d"``,
        ``"moons"``, ``"helix"``, ``"eye"``, ``"heter"``.
    seed : int, optional
        If given, seeds NumPy's global RNG for reproducible draws.

    Returns
    -------
    X_train, y_train, X_test, y_test, X_valid, y_valid : ndarray
        Three independent draws from the same scenario.
    """
    if scenario not in _SCENARIOS:
        raise NotImplementedError(
            "Unknown scenario {!r}; choose from {}".format(
                scenario, sorted(_SCENARIOS)))

    if seed is not None:
        np.random.seed(seed)

    generate = _SCENARIOS[scenario]
    X_train, y_train = generate(n_instance)
    X_test, y_test = generate(n_instance)
    X_valid, y_valid = generate(n_instance)

    return X_train, y_train, X_test, y_test, X_valid, y_valid
