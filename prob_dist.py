import numpy as np

# 1 input
def y_values(X_array, y_array, value, bounds):
    y_array = y_array.flatten()
    X_array = X_array.flatten()

    order = X_array.argsort()
    y_array = y_array[order]
    X_array = X_array[order]
    all_xs = X_array[np.where(np.logical_and(X_array>=(value-bounds), X_array<=(value+bounds)))]

    all_y = []
    for near in all_xs:
        single_y = y_array[X_array.searchsorted(near, 'left')-1]
        all_y.append(single_y)

    return np.asarray(all_y)


# 2 inputs
def y2_values(X_array, X2_array, y_array, value, bounds):
    y_array = y_array.flatten()
    X_array = X_array.flatten()
    X2_array = X2_array.flatten()

    order = X_array.argsort()
    y_array = y_array[order]
    X_array = X_array[order]
    X2_array = X2_array[order]

    all_xs = X_array[np.where(np.logical_and(X_array>=(value-bounds), X_array<=(value+bounds)))]

    all_x2s = []
    for near in all_xs:
        single_x2 = X2_array[X_array.searchsorted(near, 'left')-1]
        all_x2s.append(single_x2)

    all_x2s = np.asarray(all_x2s)

    fixed_x2s = all_x2s[np.where(np.logical_and(all_x2s>=(value-bounds), all_x2s<=(value+bounds)))]

    all_y = []
    for near in fixed_x2s:
        single_y = y_array[X2_array.searchsorted(near, 'left')-1]
        all_y.append(single_y)

    return np.asarray(all_y)
