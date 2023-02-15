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

    if n == 1:
        return (np.ones(1), A)

    for k in range(n - 1):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])
    
    L: np.ndarray = np.tril(A, -1)
    U: np.ndarray = np.triu(A, 0)
    for i in range(n):
        L[i, i] = 1.0

    return (L, U)




