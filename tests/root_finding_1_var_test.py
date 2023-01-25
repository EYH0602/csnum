import unittest
import jax.numpy as jnp
from numerical_methods.root_finding_1_var import (
    Steffensen,
    Fixed_Point,
    Newton,
    Bisection,
    Secant,
)


def f(x):
    return x**3 + 4 * x**2 - 10


def test_res(xs):
    root = 1.36523
    threshold = 1e-4
    return jnp.abs(xs[-1] - root) <= threshold


class TestRoot1Var(unittest.TestCase):
    def test_Fixed_Point(self):
        def g(x):
            return x - (x**3 + 4 * x**2 - 10) / (
                3 * x**2 + 8 * x
            )

        xs = Fixed_Point(f, 1.5, g=g)
        self.assertTrue(test_res(xs))

    def test_Newton(self):
        self.assertTrue(test_res(Newton(f, 1.5)))

    def test_Secant(self):
        self.assertTrue(test_res(Secant(f, (1.5, 1))))

    def test_Steffensen(self):
        self.assertTrue(test_res(Steffensen(f, 1.5)))


if __name__ == "__main__":
    unittest.main()
