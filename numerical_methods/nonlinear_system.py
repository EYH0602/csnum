import jax.numpy as jnp
from jax import grad
import numpy as np
from typing import Callable, List, Optional
from numerical_methods.root_finding_1_var import general_iter_method


# G : R^n -> R^n
NCts = Callable[[List[float]], List[float]]  # not true, but works for know


def _converged(xs, ys, tol):
    xs = np.array(xs)
    ys = np.array(ys)
    return np.linalg.norm(xs - ys) < tol


def _ndapply(fs: NCts, xs: List[float]) -> List[float]:
    return [f(xs) for f in fs]


def fixed_point(gs: NCts, p0: List[float], tol=1e-4, max_iter=15):
    return general_iter_method(
        lambda ps: _ndapply(gs, ps[-1]),
        p0,
        tol,
        max_iter,
        _converged,
        method="Fixed Point Method",
        return_all=True,
    )
