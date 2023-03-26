import jax.numpy as jnp
from jax import grad
import numpy as np
from typing import Callable, List, Optional, Iterable
from inspect import signature

from csnum.root_finding_1_var import general_iter_method


# G : R^n -> R^n
# fix: f(x1, x2, ...) instead of f([x1, x2, ...])
NCts = Callable[[List[float]], List[float]]  # not true, but works for know


def _converged(xs, ys, tol):
    xs = np.array(xs)
    ys = np.array(ys)
    return np.linalg.norm(xs - ys) < tol


def _ndapply(fs: NCts, xs: List[float]) -> List[float]:
    return jnp.array([f(*xs) for f in fs])


def fixed_point(
    fs: NCts, p0: List[float], tol=1e-4, max_iter=15, gs: Optional[NCts] = None
):
    if gs is None:
        gs = [lambda x: x - f(x) for f in fs]

    return general_iter_method(
        lambda ps: _ndapply(gs, ps[-1]),
        p0,
        tol,
        max_iter,
        _converged,
        method="Fixed Point Method",
        return_all=True,
    )


def jacobian(fs: NCts) -> List[NCts]:
    J = []
    for f in fs:
        sig = signature(f)
        J.append([grad(f, i) for i in range(len(sig.parameters))])
    return J


def jacobian_apply(J: List[NCts], xs: Iterable[float]) -> np.matrix:
    return jnp.array([_ndapply(fs, xs) for fs in J])


def newton(fs: NCts, p0: Iterable[float], tol=1e-4, max_iter=15, Jf=None):
    if Jf is None:
        Jf = jacobian(fs)

    def succ(ps):
        xs = ps[-1]
        J = jacobian_apply(Jf, xs)
        F = _ndapply(fs, xs)
        ys = jnp.linalg.solve(J, F)
        return xs - ys

    return general_iter_method(
        succ,
        p0,
        tol,
        max_iter,
        _converged,
        method="Newton's Method",
        return_all=True,
    )
