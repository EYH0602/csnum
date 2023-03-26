import numpy as np
from numpy.linalg import norm
from typing import Tuple, Callable, Union

from csnum.linear_direct_methods import lu_factorization, lu_solve


def _converged(curr: np.ndarray, prev: np.ndarray, thresh: float):
    return norm(curr - prev, ord=np.inf) / norm(curr, ord=np.inf) <= thresh


def general_iter_method(
    succ: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int,
    thresh: float,
    return_iter: bool,
) -> Union[Tuple[np.ndarray, int], np.ndarray]:
    """General Iterative Method for linear systems

    Args:
        succ (Callable[[np.ndarray], np.ndarray]): compute the next approximation based on current solution
        x0 (np.ndarray): initial guess
        max_iter (int): maximum allowed iteration
        thresh (float): convergence threshold
        return_iter (bool): if True, return #iteration with the solution

    Returns:
        Union[Tuple[np.ndarray, int], np.ndarray]: solution w/o #iteration
    """
    for it in range(max_iter + 1):
        x = succ(x0)
        if _converged(x, x0, thresh):
            return (x, it) if return_iter else x
        x0, x = x, x0
    return (x, max_iter) if return_iter else x


def jacobi(
    A: np.matrix,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    thresh: float = 1e-8,
    return_iter: bool = False,
):
    def succ(x0):
        x = np.zeros(shape=x0.shape)
        for i in range(x0.shape[0]):
            x[i] = (-A[i, :i] @ x0[:i] - A[i, i + 1 :] @ x0[i + 1 :] + b[i]) / A[i, i]
        return x

    return general_iter_method(succ, x0, max_iter, thresh, return_iter)


def gauss_seidel(
    A: np.matrix,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    thresh: float = 1e-8,
    return_iter: bool = False,
):
    def succ(x0):
        x = np.zeros(shape=x0.shape)
        for i in range(x0.shape[0]):
            x[i] = (-A[i, :i] @ x[:i] - A[i, i + 1 :] @ x0[i + 1 :] + b[i]) / A[i, i]
        return x

    return general_iter_method(succ, x0, max_iter, thresh, return_iter)


def sor(
    A: np.matrix,
    b: np.ndarray,
    x0: np.ndarray,
    w: float = 1.1,
    max_iter: int = 10,
    thresh: float = 1e-8,
    return_iter: bool = False,
):
    def succ(x0):
        x = np.zeros(shape=x0.shape)
        for i in range(x0.shape[0]):
            gs_update = -A[i, :i] @ x[:i] - A[i, i + 1 :] @ x0[i + 1 :] + b[i]
            x[i] = (1 - w) * x0[i, :] + w * gs_update / A[i, i]
        return x

    return general_iter_method(succ, x0, max_iter, thresh, return_iter)


def refinement(
    A: np.matrix,
    b: np.ndarray,
    factorization=lu_factorization,
    max_iter: int = 10,
    thresh: float = 1e-8,
    return_iter: bool = False,
):
    def refine(p: Tuple[np.ndarray, np.ndarray]):
        L, U = p
        xk = lu_solve(L, U, b)

        def succ(xk):
            r = b - A @ xk
            y = lu_solve(L, U, r)
            return xk + y

        return general_iter_method(succ, xk, max_iter, thresh, return_iter)

    return factorization(A).map(refine)
