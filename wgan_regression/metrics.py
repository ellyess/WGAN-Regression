"""Quantitative metrics for comparing generated and true distributions.

The models in this project produce *samples*, not point predictions, so
they are evaluated as distributions:

- :func:`conditional_wasserstein`: how well does the model reproduce the
  conditional p(y | x)? Measured as the 1-D Wasserstein-1 distance between
  true and generated y values inside thin slices around chosen x values,
  averaged over slices.
- :func:`mmd_rbf`: how well does the model reproduce the *joint* p(x, y)?
  Measured as the kernel Maximum Mean Discrepancy between the two sample
  sets, with an RBF kernel and the median-distance bandwidth heuristic.
- :func:`kde_nll`: average negative log-likelihood of held-out true
  samples under a kernel density estimate fitted to the generated
  samples. Penalises both missing modes and spurious ones.

All functions are NumPy/SciPy/scikit-learn only, so they work with any
model backend (TensorFlow, PyTorch, GPy).
"""

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity


def _slice_mask(X, value, bounds):
    """Boolean mask of rows whose first column lies in ``value +- bounds``."""
    x = np.asarray(X).reshape(len(X), -1)[:, 0]
    return (x >= value - bounds) & (x <= value + bounds)


def conditional_wasserstein(X_true, y_true, X_gen, y_gen, x_values=None,
                            bounds=None, min_samples=5):
    """Mean Wasserstein-1 distance between conditional slices of y.

    For each ``x`` in ``x_values``, collects the true and generated y
    values whose input lies within ``x +- bounds`` and computes the 1-D
    Wasserstein-1 (earth mover's) distance between the two sets. Slices
    where either side has fewer than ``min_samples`` points are skipped.

    Parameters
    ----------
    X_true, y_true : array-like
        True input and output samples.
    X_gen, y_gen : array-like
        Generated input and output samples.
    x_values : sequence of float, optional
        Slice centres. Defaults to the 10th to 90th percentiles of the
        true inputs in steps of 20.
    bounds : float, optional
        Slice half-width. Defaults to 2.5% of the true input range.
    min_samples : int, optional
        Minimum points per side for a slice to count.

    Returns
    -------
    float
        Mean W1 across evaluated slices (np.nan if none were evaluable).
    """
    X_true = np.asarray(X_true)
    X_gen = np.asarray(X_gen)
    y_true = np.asarray(y_true).flatten()
    y_gen = np.asarray(y_gen).flatten()

    x_first = X_true.reshape(len(X_true), -1)[:, 0]
    if x_values is None:
        x_values = np.percentile(x_first, [10, 30, 50, 70, 90])
    if bounds is None:
        bounds = 0.025 * (x_first.max() - x_first.min())

    distances = []
    for value in x_values:
        true_slice = y_true[_slice_mask(X_true, value, bounds)]
        gen_slice = y_gen[_slice_mask(X_gen, value, bounds)]
        if len(true_slice) >= min_samples and len(gen_slice) >= min_samples:
            distances.append(wasserstein_distance(true_slice, gen_slice))

    return float(np.mean(distances)) if distances else float("nan")


def mmd_rbf(samples_a, samples_b, bandwidth=None):
    """Kernel Maximum Mean Discrepancy between two sample sets.

    Uses an RBF kernel; if ``bandwidth`` is not given, the median pairwise
    distance across the pooled samples is used (the standard heuristic).
    Returns the biased MMD^2 estimate; 0 means the kernel mean embeddings
    coincide, larger means more distinguishable distributions.

    Parameters
    ----------
    samples_a, samples_b : array-like of shape (n, d)
        The two sample sets (e.g. true and generated joint (x, y) points).
    bandwidth : float, optional
        RBF kernel bandwidth (sigma).

    Returns
    -------
    float
        The MMD^2 estimate.
    """
    a = np.asarray(samples_a, dtype=float).reshape(len(samples_a), -1)
    b = np.asarray(samples_b, dtype=float).reshape(len(samples_b), -1)

    def sq_dists(u, v):
        return ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)

    d_aa, d_bb, d_ab = sq_dists(a, a), sq_dists(b, b), sq_dists(a, b)

    if bandwidth is None:
        pooled = np.concatenate([
            d_aa[np.triu_indices_from(d_aa, k=1)],
            d_bb[np.triu_indices_from(d_bb, k=1)],
            d_ab.ravel(),
        ])
        median_sq = np.median(pooled)
        bandwidth = np.sqrt(median_sq / 2) if median_sq > 0 else 1.0

    gamma = 1.0 / (2 * bandwidth ** 2)
    k_aa = np.exp(-gamma * d_aa).mean()
    k_bb = np.exp(-gamma * d_bb).mean()
    k_ab = np.exp(-gamma * d_ab).mean()

    return float(k_aa + k_bb - 2 * k_ab)


def kde_nll(true_samples, gen_samples, bandwidth=None):
    """Mean negative log-likelihood of true samples under a KDE of the
    generated samples.

    A Gaussian KDE is fitted to the generated samples (Scott's-rule
    bandwidth unless given) and evaluated at the held-out true samples.
    Lower is better; a model that misses a data mode entirely is heavily
    penalised because true points in that mode fall in near-zero density.

    Parameters
    ----------
    true_samples : array-like of shape (n, d)
        Held-out true samples.
    gen_samples : array-like of shape (m, d)
        Generated samples the density is fitted to.
    bandwidth : float, optional
        KDE bandwidth override.

    Returns
    -------
    float
        Mean NLL over the true samples.
    """
    true = np.asarray(true_samples, dtype=float).reshape(len(true_samples), -1)
    gen = np.asarray(gen_samples, dtype=float).reshape(len(gen_samples), -1)

    if bandwidth is None:
        # Scott's rule with a floor to survive near-degenerate sample sets.
        n, d = gen.shape
        spread = gen.std(axis=0).mean()
        bandwidth = max(spread * n ** (-1.0 / (d + 4)), 1e-3)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(gen)
    return float(-kde.score_samples(true).mean())
