import unittest
import numpy as np

from numerical_methods.nonlinear_system import fixed_point


def g1(xs):
    x1, x2 = xs
    return (x1**2 + x2**2 + 8) / 10


def g2(xs):
    x1, x2 = xs
    return (x1 * x2**2 + x1 + 8) / 10


G = [g1, g2]


class TestNonlinearSystem(unittest.TestCase):
    def test_fixed_point(self):
        sol = fixed_point(G, [0, 0], tol=1e-6)[-1]
        self.assertTrue(np.allclose(np.array(sol), np.ones(shape=(2,)), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
