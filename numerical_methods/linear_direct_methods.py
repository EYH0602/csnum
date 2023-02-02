import jax.numpy as jnp
import numpy as np
from returns.maybe import Maybe, Nothing, Some
from typing import Tuple


def _select_p(A: np.matrix, i: int, pivot: str) -> Maybe[int]:
    """compute p for gaussian elimination

    Args:
        A (np.matrix): current matrix
        i (int): iteration
        pivot (str): pivot policy

    Returns:
        Maybe[int]: p
    """
    xs = A[i:, i]
    p = i + 1

    match pivot:
        case "none":
            # find smallest non-zero entry
            idx = np.nonzero(xs)[0]
            # if no integer p can be found
            if len(idx) == 0:
                return Nothing
            p = np.min(idx) + i  # NOTE i is the offset
        case "partial":
            # find max entry
            p = np.argmax(xs)
        case _:
            return Nothing

    return Some(p)


def gaussian_elimination(
    A: np.matrix, pivot: str = "none"
) -> Maybe[Tuple[np.matrix, np.matrix]]:
    """Gaussian Elimination without backward substitution
    Args:
        A (np.matrix): matrix to do gaussian elimination
        pivot (str, optional): pivot policy in ["none", "partial"]. Defaults to "none".

    Returns:
        Maybe[Tuple[np.matrix, np.matrix]]: (elimination result, multipliers used)
    """

    if A is None:
        return Nothing

    A = np.ndarray.copy(A)
    n = A.shape[0]
    m = np.zeros(A.shape)

    # elimination process
    for i in range(n - 1):

        # search for valid p based on pivoting
        p = i + 1
        match _select_p(A, i, pivot):
            case Some(next_p):
                p = next_p
            case Nothing:
                return Nothing
        # no unique solution
        if A[p, i] == 0:
            return Nothing
        # swap
        if i != p:
            A[[p, i]] = A[[i, p]]

        # column elimination
        for j in range(i + 1, n):
            m[j, i] = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m[j, i] * A[i, :]

    # no unique solution
    if A[-1, -1] == 0:
        return Nothing

    return Some((A, m))


def back_substitution(mat: np.matrix) -> np.array:
    """back substitution step of gaussian elimination,
    this is assumed to be used upon success of `gaussian_elimination`

    Args:
        mat (np.matrix): matrix from result of gaussian elimination

    Returns:
        np.array: solved values
    """

    x = np.zeros((mat.shape[0], 1))
    n = mat.shape[1]
    # split input mat to make indexing easier
    A = mat[:, :-1]
    b = mat[:, -1]

    x[-1, :] = b[-1, :] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i, :] = (b[i, :] - A[i, i + 1 :] @ x[i + 1 :, :]) / A[i, i]
    return x


def lu_decomposition(A: np.matrix) -> Maybe[Tuple[np.matrix, np.matrix]]:
    """LU Decomposition

    Args:
        A (np.matrix): square matrix to decompose

    Returns:
        Maybe[Tuple[np.matrix, np.matrix]]: (L, U)
    """
    if A is None:
        return Nothing
    # only decompose square matrix
    if A.shape[0] != A.shape[1]:
        return Nothing
    return gaussian_elimination(A, pivot="none")


def solve(A, b):
    pass
