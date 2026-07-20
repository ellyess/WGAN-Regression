"""Conditional-slice helpers for probability density plots.

To compare how well each model captures the conditional distribution
p(y | x), we take a thin vertical slice of the data around a chosen input
value and collect the y values that fall inside it. Kernel density
estimates of these slices (true data vs WGAN samples vs GPR samples) give
the density comparison plots used in the paper and notebooks.
"""

import numpy as np


def y_values(X_array, y_array, value, bounds):
    """Collect y values whose x lies within ``value ± bounds``.

    For single-input datasets.

    Parameters
    ----------
    X_array, y_array : array-like
        Paired input and output values (flattened internally).
    value : float
        Centre of the slice on the x axis.
    bounds : float
        Half-width of the slice.

    Returns
    -------
    ndarray
        The y values of all points inside the slice.
    """
    y_array = y_array.flatten()
    X_array = X_array.flatten()

    order = X_array.argsort()
    y_array = y_array[order]
    X_array = X_array[order]
    in_slice = np.logical_and(X_array >= (value - bounds),
                              X_array <= (value + bounds))
    all_xs = X_array[np.where(in_slice)]

    all_y = []
    for near in all_xs:
        single_y = y_array[X_array.searchsorted(near, 'left') - 1]
        all_y.append(single_y)

    return np.asarray(all_y)


def y2_values(X_array, X2_array, y_array, value, bounds):
    """Collect y values where *both* inputs lie within ``value ± bounds``.

    Two-input version of :func:`y_values` for the ``3d`` and ``helix``
    datasets: a point contributes its y only if x1 and x2 both fall inside
    the slice.
    """
    y_array = y_array.flatten()
    X_array = X_array.flatten()
    X2_array = X2_array.flatten()

    order = X_array.argsort()
    y_array = y_array[order]
    X_array = X_array[order]
    X2_array = X2_array[order]

    in_slice = np.logical_and(X_array >= (value - bounds),
                              X_array <= (value + bounds))
    all_xs = X_array[np.where(in_slice)]

    all_x2s = []
    for near in all_xs:
        single_x2 = X2_array[X_array.searchsorted(near, 'left') - 1]
        all_x2s.append(single_x2)

    all_x2s = np.asarray(all_x2s)

    fixed_x2s = all_x2s[np.where(np.logical_and(all_x2s >= (value - bounds),
                                                all_x2s <= (value + bounds)))]

    all_y = []
    for near in fixed_x2s:
        single_y = y_array[X2_array.searchsorted(near, 'left') - 1]
        all_y.append(single_y)

    return np.asarray(all_y)
