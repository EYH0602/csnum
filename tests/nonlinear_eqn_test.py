import unittest
import jax.numpy as jnp
from csnum.nonlinear_eqn import (
    steffensen,
    fixed_point,
    newton,
    bisection,
    secant,
)


def f(x):
    return x**3 + 4 * x**2 - 10


def test_res(xs):
    root = 1.36523
    threshold = 1e-4
    return jnp.abs(xs[-1] - root) <= threshold


class TestRoot1Var(unittest.TestCase):
    def test_fixed_point(self):
        def g(x):
            return x - (x**3 + 4 * x**2 - 10) / (3 * x**2 + 8 * x)

        xs = fixed_point(f, 1.5, g=g)
        self.assertTrue(test_res(xs))

    def test_newton(self):
        self.assertTrue(test_res(newton(f, 1.5)))

    def test_secant(self):
        self.assertTrue(test_res(secant(f, (1.5, 1))))

    def test_steffensen(self):
        self.assertTrue(test_res(steffensen(f, 1.5)))


if __name__ == "__main__":
    unittest.main()
