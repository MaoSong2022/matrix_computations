from typing import Callable
import numpy as np


def forward_difference(sequence: np.ndarray) -> np.ndarray:
    return np.diff(sequence)


def aitken_method(sequence: np.ndarray) -> np.ndarray:
    """
    Use Aitken method to obtain a sequence with a faster convergence rate
    for a given convergent sequence.

    Args:
        sequence (np.ndarray): The input convergent sequence.

    Returns:
        np.ndarray: The result of sequence with a faster convergence rate.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Section 2.5
    """
    result = np.zeros(len(sequence) - 2)
    diff = forward_difference(sequence)
    ddiff = forward_difference(diff)
    for i in range(len(sequence) - 2):
        result[i] = sequence[i] - diff[i] ** 2 / ddiff[i]
    return result


def steffensen_method(
    f: Callable[[float], float], x0: float, tol: float = 1e-6
) -> float:
    """
    Calculate the fix point of f(x) using the steffensen method.

    Args:
        f (Callable[float, float]): The function for which to find the fix point.
        x0 (float): The initial guess for the root.
        tol (float, optional): The tolerance for the approximation. Defaults to 1e-6.

    Returns:
        float: The approximate root of the function.

    Raises:
        ValueError: If the function is not differentiable.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.6
    """
    x1 = f(x0)
    x2 = f(x1)
    if x2 - 2 * x1 + x0 == 0:
        raise ValueError("The function is not differentiable.")
    x = x0 - ((x1 - x0) ** 2) / (x2 - 2 * x1 + x0)
    while abs(x - x0) > tol:
        x0 = x
        x1 = f(x0)
        x2 = f(x1)
        if x2 - 2 * x1 + x0 == 0:
            raise ValueError("The function is not differentiable.")
        x = x0 - ((x1 - x0) ** 2) / (x2 - 2 * x1 + x0)

    return x
