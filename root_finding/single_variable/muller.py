from typing import Callable


def muller_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    x2: float,
    tol: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """
    Muller's Method is an iterative root-finding algorithm that finds the root of a given function within a specified tolerance and maximum number of iterations.

    Parameters:
        - f: A callable object that represents the function for which we want to find the root.
        - x0: The initial guess for the first point.
        - x1: The initial guess for the second point.
        - x2: The initial guess for the third point.
        - tol: The tolerance for the root approximation. Defaults to 1e-6.
        - max_iterations: The maximum number of iterations to perform. Defaults to 100.

    Returns:
        - The approximate root of the function within the specified tolerance.

    Raises:
        - ValueError: If the maximum number of iterations is reached without finding a root within the specified tolerance.

    References:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.8
    """
    h1 = x1 - x0
    h2 = x2 - x1
    delta1 = (f(x1) - f(x0)) / h1
    delta2 = (f(x2) - f(x1)) / h2
    d = (delta2 - delta1) / (h2 + h1)
    iters = 3

    while iters <= max_iterations:
        b = delta2 + h2 * d
        D = (b**2 - 4 * f(x2) * d) ** 0.5  # maybe complex

        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b = D

        h = -2 * f(x2) / E
        x = x2 + h
        print(f"Iteration {iters}: x = {x:.2f}, f(x) = {f(x):.2f}")

        if abs(h) < tol:
            return x

        x0 = x1
        x1 = x2
        x2 = x
        h1 = x1 - x0
        h2 = x2 - x1
        delta1 = (f(x1) - f(x0)) / h1
        delta2 = (f(x2) - f(x1)) / h2
        d = (delta2 - delta1) / (h2 + h1)
        iters += 1

    raise ValueError("Maximum number of iterations reached.")
