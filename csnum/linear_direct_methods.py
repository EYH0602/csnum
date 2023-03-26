import numpy as np
from returns.maybe import Maybe, Nothing, Some
from returns.result import Result, Success, Failure
from typing import Tuple


def _select_p(A: np.ndarray, i: int, pivot: str) -> Maybe[Tuple[int, int]]:
    """compute p for gaussian elimination

    Args:
        A (np.matrix): current matrix
        i (int): iteration
        pivot (str): pivot policy

    Returns:
        Maybe[Tuple[int, int]]: p
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
            p = min(idx) + i  # NOTE i is the offset
        case "partial":
            # find max entry
            p = np.argmax(xs, axis=0) + i
        case "scaled_partial":
            s = np.max(np.abs(A[i, :]))  # scale factor
            if s == 0:
                return Nothing
            p = int(np.argmax(xs / s)) + i
        case "complete":
            A_sub = A[i:, i:]
            p, q = np.unravel_index(np.argmax(A_sub), A_sub.shape)  # type: ignore
            p += i
            q += i - 1
        case _:
            return Nothing

    return Some((p, q))


def gaussian_elimination(
    A: np.ndarray, pivot: str = "none"
) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """_summary_

    Args:
        A (np.ndarray): matrix to do gaussian elimination
        pivot (str, optional): pivot policy in ["none", "partial", "scaled_partial", "complete"].
        Defaults to "none".

    Returns:
        Result[Tuple[np.ndarray, np.ndarray], str]: (elimination result, multipliers used) or a failure message
    """

    if A is None:
        return Failure("A is None")

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
                return Failure("Factorization impossible")
        # no unique solution
        if A[p, i] == 0:
            return Failure("Factorization impossible")
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
        return Failure("A is singular")

    return Success((A, m))


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """back substitution step of gaussian elimination,
    this is assumed to be used upon success of `gaussian_elimination`

    Args:
        mat (np.ndarray): matrix from result of gaussian elimination

    Returns:
        np.array: solved values
    """

    n = A.shape[0]
    x = np.zeros(shape=b.shape)

    x[-1, :] = b[-1, :] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i, :] = (b[i, :] - A[i, i + 1 :] @ x[i + 1 :, :]) / A[i, i]

    return x


def forward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """forward substitution step,
    this is assumed to be used upon success of `gaussian_elimination`

    Args:
        mat (np.ndarray): a square matrix

    Returns:
        np.array: solved values
    """

    n = A.shape[0]
    x = np.zeros(shape=b.shape)

    x[0, :] = b[0, :] / A[0, 0]
    for i in range(1, n):
        x[i, :] = (b[i, :] - A[i, :i] @ x[:i, :]) / A[i, i]

    return x


def lu_factorization(A: np.ndarray) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """LU factorization

    Args:
        A (np.matrix): square matrix to factorize

    Returns:
        Result[Tuple[np.ndarray, np.ndarray], str]: (L, U)
    """
    if A is None:
        return Failure("A is None")
    # only decompose square matrix
    if A.shape[0] != A.shape[1]:
        return Failure("A is not square matrix")
    return gaussian_elimination(A, pivot="none").map(lambda p: p[::-1])


def gauss_solve(A: np.ndarray, b: np.ndarray, pivot: str = "none"):
    """solve linear system Ax = b by gaussian elimination and back substitution.
    result depends on the success of gaussian_elimination

    Args:
        A (np.matrix): coefficient matrix
        b (np.array): value vector
        pivot (str, optional): pivot policy in ["none", "partial", "scaled_partial", "complete"].
            Defaults to "none".


    Returns:
        Result[np.array, str]: result or failure message
    """
    return gaussian_elimination(np.hstack((A, b)), pivot=pivot).map(
        lambda p: back_substitution(p[0][:, :-1], p[0][:, [-1]])
    )


def lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def is_positive_definite(x: np.ndarray) -> bool:
    return all(np.linalg.eigvals(x) > 0)


def ldl_factorization(A: np.ndarray) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """ldl factorization

    Args:
        A (np.matrix): positive definite matrix to factorize

    Returns:
        Maybe[Tuple[np.matrix, np.matrix]]: L, D
    """
    if A is None:
        return Failure("A is None")
    if not is_positive_definite(A):
        return Failure("A is not positive definite")

    n = A.shape[0]
    L = np.identity(n)
    d = np.zeros(shape=(A.shape[0], 1))
    v = np.zeros(shape=(A.shape[0], 1))

    for i in range(n):
        # for j in 1..i-1, set v_i = l_ij * d_j
        v[:i, :] = np.transpose(L[i, :i] * d[:i].T)
        # d_i = a_ii - sum l_ij v_j
        d[i] = A[i, i] - L[i, :i] @ v[:i, :]
        # for j in i+1..n, set l_ji = (a_ji - sum l_jk v_k) / d_i
        L[i + 1 :, [i]] = (A[i + 1 :, [i]] - L[i + 1 :, :i] @ v[:i, :]) / d[i]

    D = np.zeros(shape=A.shape)
    np.fill_diagonal(D, d)
    return Success((L, D))


def cholesky_factorization(A: np.ndarray):
    def decomp_D(ms: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        L, D = ms
        return L @ np.power(D, 1 / 2)

    return ldl_factorization(A).map(decomp_D)


def tridiag_ones(n: int) -> np.ndarray:
    """get a tridiagonal matrix with only 1s

    Args:
        n (int): dimension

    Returns:
        np.matrix: tridiagonal matrix with only 1s
    """
    a = np.ones(n - 1)
    b = np.ones(n)
    c = np.ones(n - 1)
    return np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)


def is_tridiag(A: np.matrix) -> bool:
    B = tridiag_ones(A.shape[0])
    return all(np.where(A == 0)[0] == np.where(B == 0)[0])


def crout_factorization(A: np.matrix) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """Crout Factorization for Tridiagonal Linear Systems

    Args:
        A (np.ndarray): Tridiagonal square matrix

    Returns:
        Maybe[Tuple[np.ndarray, np.ndarray]]: (L, U)
    """
    if A.shape[0] != A.shape[1]:
        return Failure("A is not square matrix")

    if not is_tridiag(A):
        return Failure("A is not tridiagonal matrix")

    n = A.shape[0]
    L = np.zeros(shape=A.shape)
    U = np.identity(n)

    L[0, 0] = A[0, 0]
    U[0, 1] = A[0, 1] / L[0, 0]

    for i in range(1, n - 1):
        L[i, i - 1] = A[i, i - 1]
        L[i, i] = A[i, i] - L[i, i - 1] * U[i - 1, i]
        U[i, i + 1] = A[i, i + 1] / L[i, i]

    L[-1, -2] = A[-1, -2]
    L[-1, -1] = A[-1, -1] - L[-1, -2] * U[-2, -1]

    return Success((L, U))


def crout_solve(L: np.matrix, U: np.matrix, b: np.ndarray) -> np.ndarray:
    """solve for Crout Factorization

    Args:
        L (np.matrix): L output of `crout_factorization`
        U (np.matrix): U output of `crout_factorization`
        b (np.array): RHS of linear system

    Returns:
        np.array: results
    """
    z = np.zeros(shape=b.shape)
    n = b.shape[0]
    z[0] = b[0] / L[0, 0]

    # solve for Lz = b
    # don't know why, but the book use z here instead of y
    for i in range(1, n - 1):
        z[i] = (b[i] - L[i, i - 1] * z[i - 1]) / L[i, i]
    z[-1] = (b[-1] - L[-1, -2] * z[-2]) / L[-1, -1]
    # solve for Ux = z
    x = z.copy()
    for i in range(n - 2, -1, -1):
        x[i] = z[i] - U[i, i + 1] * x[i + 1]

    return x
