import numpy as np


def lagrangian_interpolation(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Generates the Lagrangian interpolation for a set of data points.

    Parameters:
    - data (np.ndarray): An array of shape (n, 2) representing the data points.
    - x (np.ndarray): An array of shape (m,) representing the points to interpolate.

    Returns:
    - np.ndarray: An array of shape (m,) representing the interpolated values at the given points.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Theorem 3.2
    """
    result = np.zeros_like(x)
    n = data.shape[0]
    for i in range(n):
        numerator, dominator = 1.0, 1.0
        for j in range(n):
            if j == i:
                continue
            numerator = numerator * (x - data[j, 0])
            dominator = dominator * (data[i, 0] - data[j, 0])

        L_i = numerator / dominator
        result = result + data[i, 1] * L_i

    return result
