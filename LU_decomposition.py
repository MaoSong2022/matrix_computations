import numpy as np
from typing import Tuple


def gaussian_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use the gaussian transformation to compute the LU decompositions of A

    Args:
        A (np.ndarray): an square invertible matrix of size n-by-n

    Raises:
        ValueError: raises if A is not a square matrix.
        ZeroDivisionError: raises if A is not invertible

    Returns:
        Tuple[np.ndarray, np.ndarray]: LU decomposition of matrix A
        L (np.ndarray): an square unit lower triangular matrix  of size n-by-n.
        U (np.ndarray): an square upper triangular matrix  of size n-by-n.

    Reference:
        <<Matrix Computations>> 4-th Edition, Section 3.2.1~3.2.6
    """
    m, n = A.shape

    if m != n:
        raise ValueError("LU decomposition is only valid for square matrix.")
    
    L = np.eye(n)
    I = np.eye(n)
    for k in range(n):
        if abs(A[k, k]) <= 1e-8:
            raise ZeroDivisionError("pivot should not be zero")
        tau = np.zeros((n, 1))
        tau[k+1:, 0] = A[k+1:, k] / A[k, k]
        M = np.eye(n) - tau @ I[:, k:k+1].T
        A = M @ A
        L += tau @ I[:, k:k+1].T

    return (L, A)

def out_product_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use the out product to compute the LU decompositions of A

    Args:
        A (np.ndarray): an square invertible matrix of size n-by-n

    Raises:
        ValueError: raises if A is not a square matrix.
        ZeroDivisionError: raises if A is not invertible

    Returns:
        Tuple[np.ndarray, np.ndarray]: LU decomposition of matrix A
        L (np.ndarray): an square unit lower triangular matrix  of size n-by-n.
        U (np.ndarray): an square upper triangular matrix  of size n-by-n.

    Reference:
        <<Matrix Computations>> 4-th Edition, Algorithm 3.2.1
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


def gaxpy_LU(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use the gaxpy method to find the LU decomposition of A

    Args:
        A (np.ndarray): an square invertible matrix of size n-by-n

    Raises:
        ValueError: raises if A is not a square matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: LU decomposition of matrix A
        L (np.ndarray): an square unit lower triangular matrix  of size n-by-n.
        U (np.ndarray): an square upper triangular matrix  of size n-by-n.

    Reference:
        <<Matrix Computations>> 4-th Edition, Algorithm 3.2.2
    """
    m, n = A.shape

    if m != n:
        raise ValueError("LU decomposition is only valid for square matrix.")

    L: np.ndarray = np.eye(n)
    U: np.ndarray = np.zeros((n, n))

    v: np.ndarray = np.zeros(n)
    for i in range(n):
        if i == 0:
            v = A[:, 0: 1]
        else:
            a = A[:, i: i+1]
            z = np.linalg.solve(L[:i, :i], a[:i])
            U[:i, i:i+1] = z
            v[i:] = a[i:] - L[i:, :i] @ z
        
        U[i, i] = v[i]
        L[i + 1:, i:i+1] = v[i+1:] / v[i]

    return (L, U)
