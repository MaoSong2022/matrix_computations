from typing import Callable


def newton_method(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
) -> float:
    """
    Performs the Newton's method to find the root of a function.

    Args:
        f (Callable[[float], float]): The function for which we want to find the root.
        df (Callable[[float], float]): The derivative of the function.
        x0 (float): The initial guess for the root.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        float: The estimated root of the function.

    Raises:
        ValueError: If the derivative is zero at any point during the iteration.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.3

    """
    if df(x0) == 0:
        raise ValueError("The derivative is zero.")
    x = x0 - f(x0) / df(x0)
    while abs(x - x0) > tol:
        x0 = x
        if df(x0) == 0:
            raise ValueError("The derivative is zero.")
        x = x0 - f(x0) / df(x0)
    return x


def secant_method(
    f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-6
) -> float:
    """
    Calculate the root of a function using the secant method.

    Args:
        f (Callable[[float], float]): The function for which to find the root.
        x0 (float): The initial guess for the root.
        x1 (float): Another initial guess for the root.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Raises:
        ValueError: If the function is not differentiable.

    Returns:
        float: The estimated root of the function.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.4
    """
    if f(x0) == f(x1):
        raise ValueError("The function is not differentiable.")
    x = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
    while abs(x - x1) > tol:
        x0 = x1
        x1 = x
        if f(x1) == f(x0):
            raise ValueError("The function is not differentiable.")
        x = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
    return x


def false_position(
    f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-6
) -> float:
    """
    Calculate the root of a function using the false position method.

    Parameters:
        f (Callable[[float], float]): The function for which to find the root.
        x0 (float): The initial guess for the root.
        x1 (float): Another guess for the root.
        tol (float, optional): The tolerance for the approximation. Defaults to 1e-6.

    Returns:
        float: The approximate root of the function.

    Raises:
        ValueError: If the function is not differentiable.

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Algorithm 2.5
    """
    if f(x0) == f(x1):
        raise ValueError("The function is not differentiable.")
    x = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
    while abs(x - x1) > tol:
        if f(x) * f(x1) < 0:
            x0 = x1
        x1 = x
        if f(x1) == f(x0):
            raise ValueError("The function is not differentiable.")
        x = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
    return x


def newton_multiple_roots(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
) -> float:
    pass


if __name__ == "__main__":
    import math

    def f(x):
        return math.cos(x) - x

    x0 = 0.5
    x1 = math.pi / 4
    root = false_position(f, x0, x1)
    print(root)
