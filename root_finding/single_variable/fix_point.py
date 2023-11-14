from typing import Callable

from common import difference


def fix_point(f: Callable, x0: float, tol: float = 1e-6) -> float:
    """
    Find the fix point of a function using the fixed-point iteration method.

    Parameters:
        f (Callable): The function for which to find the fix point.
        x0 (float): The initial guess for the fix point.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        float: The fix point of the function.

    Raises:
        ValueError: If the function violates Fix-Point Theorem

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.2; Theorem 2.4
    """
    x = x0
    while abs(f(x) - x) > tol:
        if abs(difference.numerical_gradient(f, x)) > 1:
            raise ValueError("The fix point does not converge.")
        x = f(x)

    return x
