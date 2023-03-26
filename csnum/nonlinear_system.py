import jax.numpy as jnp
from jax import grad
import numpy as np
from typing import Callable, List, Optional, Iterable, TypeVar, Any
from inspect import signature

from csnum.nonlinear_eqn import general_iter_method


# G : R^n -> R^n
# ! using this to denote f(x1, x2, ..., xn) for x_i in float
# ! if anyone knows a better type notation, please let me know
NxFloat = TypeVar("NxFloat", List[float], np.ndarray, jnp.ndarray)
NCts = Callable[[NxFloat], float]  # not true, but works for know


def _converged(xs, ys, tol):
    xs = np.array(xs)
    ys = np.array(ys)
    return np.linalg.norm(xs - ys) < tol


def _ndapply(fs: List[NCts], xs: List[float]) -> List[float]:
    return [f(*xs) for f in fs]


def fixed_point(
    fs: List[NCts],
    p0: List[float],
    tol=1e-4,
    max_iter=15,
    gs: Optional[List[NCts]] = None,
):
    succ: List[NCts] = gs if gs else [lambda x: x - f(x) for f in fs]

    return general_iter_method(
        lambda ps: _ndapply(succ, ps[-1]),
        p0,
        tol,
        max_iter,
        _converged,
        method="Fixed Point Method",
        return_all=True,
    )


def jacobian(fs: List[NCts]) -> List[List[NCts]]:
    J = []
    for f in fs:
        sig = signature(f)
        J.append([grad(f, i) for i in range(len(sig.parameters))])
    return J


def jacobian_apply(J: List[List[NCts]], xs: List[float]) -> jnp.ndarray:
    return jnp.array([_ndapply(fs, xs) for fs in J])


def newton(
    fs: List[NCts],
    p0: Iterable[float],
    tol=1e-4,
    max_iter=15,
    Jf: Optional[List[List[NCts]]] = None,
):
    jacobian_mat = Jf if Jf else jacobian(fs)

    def succ(ps):
        xs = ps[-1]
        J = jacobian_apply(jacobian_mat, xs)
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
