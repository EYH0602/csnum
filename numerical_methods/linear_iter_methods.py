import numpy as np
from numpy.linalg import norm
from typing import Tuple, List, Callable


def _converged(curr: np.ndarray, prev: np.ndarray, thresh: float) -> bool:
    return norm(curr - prev, ord=np.inf) / norm(curr, ord=np.inf) <= thresh


def general_iter_method(
    succ: Callable[[np.matrix], np.matrix], x0: np.ndarray, max_iter: int, thresh: float
) -> Tuple[np.ndarray, int]:
    for it in range(max_iter):
        x = succ(x0)
        if _converged(x, x0, thresh):
            return x, it
        x0, x = x, x0
    return x, max_iter


def jacobi(
    A: np.matrix,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    thresh: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    def succ(x0):
        x = np.zeros(shape=x0.shape)
        for i in range(x0.shape[0]):
            x[i] = (-A[i, :i] @ x0[:i] - A[i, i + 1 :] @ x0[i + 1 :] + b[i]) / A[i, i]
        return x

    return general_iter_method(succ, x0, max_iter, thresh)


def gauss_seidel(
    A: np.matrix,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    thresh: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    def succ(x0):
        x = np.zeros(shape=x0.shape)
        for i in range(x0.shape[0]):
            x[i] = (-A[i, :i] @ x[:i] - A[i, i + 1 :] @ x0[i + 1 :] + b[i]) / A[i, i]
        return x

    return general_iter_method(succ, x0, max_iter, thresh)
