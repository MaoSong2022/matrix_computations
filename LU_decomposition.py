import numpy as np
from typing import Tuple


def out_product_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """use the out product to compute the LU decomposition of matrix A

    Args:
        A (np.ndarray): Objective matrix of size m by n, A is assumed to be 
        that A(1:k, 1:k) is non-singular for k = 1: n-1

    Returns:
        Tuple[np.ndarray, np.ndarray]: LU decomposition of A,
        [L, U], L is unit lower triangular, U is upper triangular.
    """
    m, n = A.shape

    if m != n:
        raise ValueError("LU decomposition is only valid for square matrix.")

    for i in range(n):
        pivot = A[i, i]
        # check if the pivot is non-zero, equivalent to A is invertible.
        if abs(pivot) <= 1e-8:
            raise ZeroDivisionError("pivot should not be zero")
        A[i+1:, i:i+1] = A[i+1:, i:i+1] / pivot
        A[i+1:, i+1:] = A[i+1:, i+1:] - A[i+1:, i:i+1] @ A[i:i+1, i+1:]
    
    L: np.ndarray = np.eye(n) + np.tril(A, -1)
    U: np.ndarray = np.triu(A)

    return (L, U)




