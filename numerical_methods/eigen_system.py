import numpy as np
from typing import List, Tuple, Callable
from returns.result import Result, Success, Failure
from returns.pointfree import map_


def _select_idx(x: np.ndarray) -> int:
    """
    Find the smallest integer p with 1 <= p  <= n and |x_p| = ||x||_inf
    """
    norm = np.linalg.norm(x, ord=np.inf)  # ||x||_inf
    x = np.abs(x)  # |x_p|
    return np.min(np.where(np.isclose(x, norm))[0])


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
        mu = y[p, 0]
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


def inverse_power_method(
    A: np.matrix,
    x: np.ndarray,
    max_iter: int = 10,
    thresh: float = 1e-4,
) -> Result[Tuple[float, np.ndarray], Tuple[str, np.ndarray]]:
    q = x.T @ A @ x / (x.T @ x)

    def update(A, x):
        # ToDo: use LU solve
        return np.linalg.solve(A - q * np.eye(A.shape[0]), x)

    def inverse_method_res(p):
        mu, x = p
        return (1 / mu + q, x)

    return map_(inverse_method_res)(
        power_method(A, x, update=update, max_iter=max_iter, thresh=thresh),
    )


# .map(
#         inverse_method_res
#     )
