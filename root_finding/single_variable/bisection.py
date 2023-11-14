from typing import Callable, Tuple


def bisection(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-6
) -> Tuple[float, float]:
    """
    Performs the bisection method to find the root of a given
    function within a specified interval.

    Parameters:
        f (Callable[[float], float]): The function for which the root is to be found.
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        tol (float, optional): The tolerance value for convergence. Defaults to 1e-6.

    Returns:
        Tuple[float, float]: The root of the function and
            the corresponding function value at the root.

    Raises:
        ValueError: If the function values at the interval bounds have the same sign,
            indicating that the bisection method fails.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.1
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Bisection method fails.")

    p = (a + b) / 2
    while abs(f(p)) > tol:
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p = (a + b) / 2
    return p, f(p)
