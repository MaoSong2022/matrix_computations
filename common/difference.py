from typing import Callable


def numerical_gradient(f: Callable[[float], float], x: float) -> float:
    """
    Calculates the numerical gradient of a function f at a point x.

    Parameters:
        f (Callable[[float], float]): The function for which the gradient
            is to be calculated.
        x (float): The point at which the gradient is to be calculated.

    Returns:
        float: The numerical gradient of the function at the point x.
    """
    h = 1e-6
    grad = (f(x + h) - f(x - h)) / (2 * h)
    return grad
