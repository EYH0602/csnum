import unittest
import numpy as np

from numerical_methods.nonlinear_system import fixed_point


class TestNonlinearSystem(unittest.TestCase):
    def test_fixed_point(self):
        def g1(xs):
            x1, x2 = xs
            return (x1**2 + x2**2 + 8) / 10

        def g2(xs):
            x1, x2 = xs
            return (x1 * x2**2 + x1 + 8) / 10

        sol = fixed_point(None, [0, 0], tol=1e-6, gs=[g1, g2])[-1]
        self.assertTrue(np.allclose(np.array(sol), np.ones(shape=(2,)), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
