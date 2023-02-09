import numpy as np
from returns.maybe import Maybe, Nothing, Some
from typing import Tuple


def _select_p(A: np.matrix, i: int, pivot: str) -> Maybe[Tuple[int, int]]:
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
    q = i

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
            p = np.argmax(xs) + i
        case "scaled_partial":
            s = np.max(np.abs(A[i, :]))  # scale factor
            if s == 0:
                return None
            p = np.argmax(xs / s) + i
        case _:
            return Nothing

    return Some((p, q))


def gaussian_elimination(
    A: np.matrix, pivot: str = "none"
) -> Maybe[Tuple[np.matrix, np.matrix]]:
    """Gaussian Elimination without backward substitution
    Args:
        A (np.matrix): matrix to do gaussian elimination
        pivot (str, optional): pivot policy in ["none", "partial", "scaled_partial"].
            Defaults to "none".

    Returns:
        Maybe[Tuple[np.matrix, np.matrix]]: (elimination result, multipliers used)
    """

    if A is None:
        return Nothing

    A = np.ndarray.copy(A)
    n = A.shape[0]
    m = np.identity(A.shape[0])

    # elimination process
    for i in range(n - 1):

        # search for valid p based on pivoting
        p = i + 1
        match _select_p(A, i, pivot):
            case Some((next_p, next_q)):
                p = next_p
                q = next_q
            case Nothing:
                return Nothing
        # no unique solution
        if A[p, i] == 0:
            return Nothing
        # swap row
        if i != p:
            A[[p, i]] = A[[i, p]]
        # swap column
        if i != q:
            A[:, [q, i]] = A[:, [i, q]]

        # column elimination
        for j in range(i + 1, n):
            m[j, i] = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m[j, i] * A[i, :]

    # no unique solution
    if A[-1, -1] == 0:
        return Nothing

    return Some((A, m))


def back_substitution(A: np.matrix, b: np.array) -> np.array:
    """back substitution step of gaussian elimination,
    this is assumed to be used upon success of `gaussian_elimination`

    Args:
        mat (np.matrix): matrix from result of gaussian elimination

    Returns:
        np.array: solved values
    """

    n = A.shape[0]
    x = np.zeros(shape=b.shape)

    x[-1, :] = b[-1, :] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i, :] = (b[i, :] - A[i, i + 1 :] @ x[i + 1 :, :]) / A[i, i]

    return x


def forward_substitution(A: np.matrix, b: np.array) -> np.array:
    """forward substitution step,
    this is assumed to be used upon success of `gaussian_elimination`

    Args:
        mat (np.matrix): a square matrix

    Returns:
        np.array: solved values
    """

    n = A.shape[0]
    x = np.zeros(shape=b.shape)

    x[0, :] = b[0, :] / A[0, 0]
    for i in range(1, n):
        x[i, :] = (b[i, :] - A[i, :i] @ x[:i, :]) / A[i, i]

    return x


def lu_factorization(A: np.matrix) -> Maybe[Tuple[np.matrix, np.matrix]]:
    """LU factorization

    Args:
        A (np.matrix): square matrix to factorize

    Returns:
        Maybe[Tuple[np.matrix, np.matrix]]: (U, L)
    """
    if A is None:
        return Nothing
    # only decompose square matrix
    if A.shape[0] != A.shape[1]:
        return Nothing
    return gaussian_elimination(A, pivot="none")


def gauss_solve(A: np.matrix, b: np.array, pivot: str = "none") -> Maybe[np.array]:
    """solve linear system Ax = b by gaussian elimination and back substitution.
    result depends on the success of gaussian_elimination

    Args:
        A (np.matrix): coefficient matrix
        b (np.array): value vector
        pivot (str, optional): pivot policy in ["none", "partial", "scaled_partial"].
            Defaults to "none".


    Returns:
        Maybe[np.array]: result
    """
    return gaussian_elimination(np.hstack((A, b)), pivot=pivot).map(
        lambda p: back_substitution(p[0][:, :-1], p[0][:, [-1]])
    )


def lu_solve(L: np.matrix, U: np.matrix, b: np.array) -> np.array:
    """solve linear system with output from LU factorization
    this is assumed to be used upon the succuss of `lu_factorization`

    Args:
        L (np.matrix): L matrix
        U (np.matrix): U matrix
        b (np.array): value vector that linear system equals

    Returns:
        np.array: solved x unknown variables
    """
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x
