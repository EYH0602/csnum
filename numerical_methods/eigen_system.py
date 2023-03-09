import numpy as np
from typing import List, Tuple, Callable
from returns.result import Result, Success, Failure
from returns.maybe import Maybe, Nothing, Some
from returns.pointfree import map_

from numerical_methods.linear_direct_methods import lu_factorization, lu_solve


def _select_idx(x: np.ndarray) -> int:
    """
    Find the smallest integer p with 1 <= p  <= n and |x_p| = ||x||_inf
    """
    norm = np.linalg.norm(x, ord=np.inf)  # ||x||_inf
    x_abs = np.abs(x)  # |x_p|
    return np.min(np.where(np.isclose(x_abs, norm))[0])


def power_method(
    A: np.matrix,
    x: np.ndarray,
    update: Callable[[np.matrix, np.ndarray], np.ndarray] = np.matmul,
    max_iter: int = 100000000,
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
    x = x / x[p, 0]
    for _ in range(max_iter):
        y = update(A, x)
        mu = y[p, 0]
        p = _select_idx(y)
        if y[p, 0] == 0:
            return Failure(
                ("A has the eigenvalue 0, select a new vector x and restart", x)
            )

        err = np.linalg.norm(x - (y / y[p, 0]), ord=np.inf)
        x = y / y[p, 0]
        if err < thresh:
            return Success((mu, x))

    # not converged
    return Failure(("The maximum number of iterations exceeded", x))


def inverse_power_method(
    A: np.matrix,
    x: np.ndarray,
    max_iter: int = 10000,
    thresh: float = 1e-4,
    q: float = 0,
) -> Result[Tuple[float, np.ndarray], Tuple[str, np.ndarray]]:

    match lu_factorization(A - q * np.eye(A.shape[0])):
        case Some((L, U)):
            inv_power_update = lambda _, x: lu_solve(L, U, x)
        case Nothing:
            return Failure(("LU Factorization failed", x))

    def inv_power_eig(p):
        mu, x = p
        return (1 / mu + q, x)

    return map_(inv_power_eig)(
        power_method(A, x, update=inv_power_update, max_iter=max_iter, thresh=thresh),
    )


def wielandt_deflation(
    A: np.matrix,
    x: np.ndarray,
    l: float,
    v: np.ndarray,
    solver=power_method,
    max_iter: int = 10000,
    thresh: float = 1e-4,
) -> Result[Tuple[float, np.ndarray], str]:
    v_abs = np.abs(v)
    i = np.min(np.where(v_abs == v_abs.max()))
    n = A.shape[0]
    B = np.zeros(shape=(n - 1, n - 1))
    if i != 0:
        for k in range(i - 1):
            for j in range(i - 1):
                B[k, j] = A[k, j] - A[i, j] * v[k] / v[i]

    if i != 0 and i != n - 1:
        for k in range(i, n - 1):
            for j in range(i - 1):
                B[k, j] = A[k + 1, j] - A[i, j] * v[k + 1] / v[i]
                B[j, k] = A[j, k + 1] - A[i, k + 1] * v[j] / v[i]

    if i != n - 1:
        for k in range(i, n - 1):
            for j in range(i, n - 1):
                B[k, j] = A[k + 1, j + 1] - A[i, j + 1] * v[k + 1] / v[i]

    match solver(B, x, max_iter=max_iter, thresh=thresh):
        case Success((mu, wp)):
            w = np.zeros(shape=v.shape)
            u = np.zeros(shape=v.shape)
            if i != 0:
                w[:i] = wp[:i]
            if i != n - 1:
                w[i + 1 :] = wp[i:]
            for k in range(n):
                u[k] = (mu - l) * w[k] + (A[i, :] @ w) * v[k] / v[i]
            return Success((mu, u))
        case Failure((msg, _)):
            return Failure(f"{solver.__name__} fails: {msg}")
