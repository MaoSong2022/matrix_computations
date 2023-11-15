from typing import Callable, Tuple

import numpy as np


def horner(coeff_P: np.ndarray, x0: float) -> np.ndarray:
    """
    Compute the evaluation of a polynomial P(x) using the Horner's method.

    Parameters:
        coeff_P (np.ndarray): The array of polynomial coefficients.
            coeff_P[i] is the coefficient of x^i
        x0 (float): The value at which the polynomial is evaluated.

    Returns:
        np.ndarray: The array of coefficients of the resulting polynomial Q(x) after the evaluation
            such that P(x) = (x - x0)Q(x) + P(x0)

    Remark:
        coeff_Q.size = coeff_P.size - 1

    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Theorem 2.19
    """
    n = coeff_P.size - 1
    coeff_Q = np.zeros(n)
    coeff_Q[-1] = coeff_P[-1]
    for i in range(n - 2, -1, -1):
        coeff_Q[i] = coeff_P[i + 1] + x0 * coeff_Q[i + 1]

    return coeff_Q


def horner_method(coeff_P: np.ndarray, x0: float) -> Tuple[float, float]:
    """
    Calculates the value of a polynomial and its derivative at a given point using the Horner's Method.

    Parameters:
        coeff_P (np.ndarray): An array containing the coefficients of the polynomial in ascending order.
            coeff_P[i] is the coefficient of x^i
        x0 (float): The value at which the polynomial and its derivative are evaluated.

    Returns:
        Tuple[float, float]: A tuple containing the value of the polynomial and its derivative at `x0`.
    """
    coeff_Q = horner(coeff_P, x0)
    n = coeff_P.size - 1

    P_x0 = coeff_P[-1]
    for i in range(n - 1, -1, -1):
        P_x0 = coeff_P[i] + x0 * P_x0

    dP_x0 = coeff_Q[-1]
    for i in range(n - 2, -1, -1):
        dP_x0 = coeff_Q[i] + x0 * dP_x0

    return P_x0, dP_x0


if __name__ == "__main__":
    coefficients = np.array([-4, 3, -3, 0, 2])
    x0 = -1.796

    # result = horner(coefficients, x0)
    result = horner_method(coefficients, x0)
    print(result)
