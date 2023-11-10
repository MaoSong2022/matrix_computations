import numpy as np
from typing import Tuple


def power_iteration(
    A: np.ndarray, max_iterations: int = 100, v: np.ndarray = None
) -> Tuple[float, np.ndarray]:
    """Find the maximum eigenvalue and corresponding eigenvector

    Args:
        A (np.ndarray): an n-by-n matrix
        max_iterations (int, optional): maximum iterations. Defaults to 100.
        v (np.ndarray): initial guess. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: the maximum eigenvalue and corresponding eigenvector.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Power iteration only supports square matrix.")

    if v is None:
        v = np.random.rand(n)

    for _ in range(max_iterations):
        v = A @ v
        norm_v = np.linalg.norm(v)
        v /= norm_v
        lamb = v.T @ A @ v

    return lamb, v


def rayleigh_quotient_iteration(
    A: np.ndarray, max_iterations: int = 100, v: np.ndarray = None
) -> Tuple[float, np.ndarray]:
    m, n = A.shape
    if m != n:
        raise ValueError("Rayleigh quotient iteration only supports square matrix.")

    if v is None:
        v = np.random.randn(n)
    elif v.ndim != 1:
        raise ValueError("The initialization vector should be 1 dimensional.")

    v = v / np.linalg.norm(v)
    lamb: float = v.T @ A @ v

    for _ in range(max_iterations):
        v = np.linalg.inv(A - lamb * np.eye(n)) @ v
        v /= np.linalg.norm(v)
        lamb = v.T @ A @ v

        # residual = A @ v - lamb * v

    return lamb, v


def orthogonal_iteration(
    A: np.ndarray, k: int = 1, max_iterations: int = 100, V: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    if m != n:
        raise ValueError("Orthogonal iteration only supports square matrix.")

    if V is not None:
        if V.shape != (n, k):
            raise ValueError(
                "The shape of initialization space does not match the input."
            )
    else:
        V = np.random.randn(n, k)

    for i in range(max_iterations):
        Q, R = np.linalg.qr(V, "reduced")
        Q /= np.linalg.norm(Q, axis=0)
        V = A @ Q

        # residual = V - Q @ eigs
    return (Q, R)
