import unittest
import numpy as np

from numerical_methods.nonlinear_system import fixed_point, jacobian, jacobian_apply


class TestNonlinearSystem(unittest.TestCase):
    def test_fixed_point(self):
        def g1(x1, x2):
            return (x1**2 + x2**2 + 8) / 10

        def g2(x1, x2):
            return (x1 * x2**2 + x1 + 8) / 10

        sol = fixed_point(None, [0, 0], tol=1e-6, gs=[g1, g2])[-1]
        self.assertTrue(np.allclose(np.array(sol), np.ones(shape=(2,)), atol=1e-4))

    def test_jacobi(self):
        def f1(x1, x2):
            return 4 * x1**2 - 20 * x1 + x2**2 / 4 + 8

        def f2(x1, x2):
            return x1 * x2**2 / 2 + 2 * x1 - 5 * x2 + 8

        F = np.array([f1, f2])

        Jf = jacobian(F)
        x0 = np.zeros(shape=(2,))

        Jx = jacobian_apply(Jf, x0)
        Jx_true = np.array([[-20, 0], [2, -5]])
        self.assertTrue(np.allclose(Jx, Jx_true, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
