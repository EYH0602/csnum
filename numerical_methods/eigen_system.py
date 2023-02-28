import numpy as np
from typing import List, Tuple, Callable
from returns.result import Result, Success, Failure


def _select_idx(vec: np.ndarray) -> int:
    return np.argmin(np.where(vec == np.linalg.norm(vec, ord=np.inf)))


def power_method(
    A: np.matrix,
    x: np.ndarray,
    update: Callable[[np.matrix, np.ndarray], np.ndarray] = np.matmul,
    max_iter: int = 10,
    thresh: float = 1e-4,
) -> Result[Tuple[float, np.ndarray], Tuple[str, np.ndarray]]:
    """Power Method to approximate the dominate eigenvalue and eigenvector

    Args:
        A (np.matrix): n x n matrix
        x (np.ndarray): nonzero n x 1 vector
        max_iter (int): maximum allowed iteration
        thresh (float): convergence threshold

    Returns:
        Result[Tuple[float, np.ndarray], Tuple[str, np.ndarray]]:
            (eigenvalue, eigenvector) or (failure message, eigenvector)
    """

    p = _select_idx(x)
    x = x / x[p]
    for _ in range(max_iter):
        y = update(A, x)
        mu = y[p]
        p = _select_idx(y)
        if y[p] == 0:
            return Failure(
                ("A has the eigenvalue 0, select a new vector x and restart", x)
            )

        err = np.linalg.norm(x - (y / y[p]), ord=np.inf)
        x = y / y[p]
        if err < thresh:
            return Success((mu, x))

    # not converged
    return Failure(("The maximum number of iterations exceeded", x))
