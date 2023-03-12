import jax.numpy as jnp
from jax import grad
from typing import Tuple, Callable, Dict, List, TypeVar
from logging import warning


Cts = Callable[[float], float]  # not true, but works for know
T = TypeVar("T")
Calcable = TypeVar("Calcable")


def Bisection(f: Cts, x: Tuple[float, float], eps=1e-4, max_iter=15):
    start, end = x
    a = [start]
    b = [end]

    def mid(a, b):
        return (a + b) / 2

    p = []
    fp = []
    for i in range(max_iter):
        p.append(mid(a[i], b[i]))
        fp.append(f(p[i]))

        if fp[i] == 0 or (b[i] - a[i]) / 2 < eps:
            p += [p[i]] * (max_iter - i - 1)
            return p

        if fp[i] * f(a[i]) > 0:
            a.append(p[i])
            b.append(b[i])
        else:
            a.append(a[i])
            b.append(p[i])

    warning(f"Bisection method fail to converge in {max_iter} iterations")
    return p


def _converged(x: float, y: float, tol: float) -> bool:
    return jnp.abs(x - y) < tol


def General_Iter_Method(
    succ: Callable[[List[Calcable]], Calcable],
    p0: Calcable,
    tol: float,
    max_iter: int,
    converged: Callable[[Calcable, Calcable, float], bool],
    method: str = "",
    return_all=True,
):
    """General Iterative Method (gim)

    Args:
        succ (Callable[List[float], float]):
            compute the new approximation p from previous approximations
        p0 (float): initial approximation
        tol (float): tolerance
        max_iter (int):
            maximum number of iterations;
            if converged before this, pad with the root it finds
        method (str, optional): method name that is calling gim

    Returns:
        List[float]: the sequence of approximations used by the method
    """
    p: List[Calcable] = []
    p.append(p0)
    for i in range(1, max_iter):
        p.append(succ(p))
        if converged(p[i], p[i - 1], tol):
            p += [p[i]] * (max_iter - i - 1)
            return p if return_all else p[-1]

    warning(f"{method} fail to converge in {max_iter} iterations")
    return p if return_all else p[-1]


def Fixed_Point(f: Cts, p0: float, tol=1e-4, max_iter=15, g: Cts = None):
    p0 = float(p0)
    if g is None:
        # f(x) = 0 --> g(x) = x
        g = lambda x: x - f(x)

    return General_Iter_Method(
        lambda ps: g(ps[-1]), p0, tol, max_iter, _converged, method="Fixed Point Method"
    )


def Newton(f: Cts, p0: float, tol=1e-4, max_iter=15):
    p0 = float(p0)

    def succ(ps):
        p = ps[-1]
        return p - f(p) / grad(f)(p)

    return General_Iter_Method(
        succ, p0, tol, max_iter, _converged, method="Newton's Method"
    )


def Secant(f: Cts, p: Tuple[float, float], eps=1e-4, max_iter=15):
    p0_org, p1_org = map(float, p)

    def succ(ps):
        if len(ps) == 1:
            return p1_org

        p0, p1 = ps[-2:]
        q0 = f(p0)
        q1 = f(p1)

        p2 = p1 - q1 * (p1 - p0) / (q1 - q0)
        return p2

    return General_Iter_Method(
        succ, p0_org, eps, max_iter, _converged, method="Secant Method"
    )


def Steffensen(f: Cts, p: float, tol=1e-4, max_iter=15, g: Cts = None):
    if g is None:
        g = lambda x: x - f(x)

    def succ(ps):
        p0 = ps[-1]
        p1 = g(p0)
        p2 = g(p1)
        return p0 - (p1 - p0) ** 2 / (p2 - 2 * p1 + p0)

    return General_Iter_Method(
        succ, p, tol, max_iter, _converged, method="Steffensen's Method"
    )
