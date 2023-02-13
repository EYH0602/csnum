import unittest
import numpy as np

from numerical_methods.linear_iter_methods import jacobi, gauss_seidel


class TestRoot1Var(unittest.TestCase):
    def test_jacobi(self):
        A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([[6, 25, -11, 15]]).T
        x0 = np.zeros(shape=b.shape)

        x, _ = jacobi(A, b, x0, max_iter=20)
        self.assertTrue(np.allclose(np.linalg.solve(A, b), x))

    def test_gauss_seidel(self):
        A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([[6, 25, -11, 15]]).T
        x0 = np.zeros(shape=b.shape)

        x, _ = gauss_seidel(A, b, x0, max_iter=20)
        self.assertTrue(np.allclose(np.linalg.solve(A, b), x))


if __name__ == "__main__":
    unittest.main()
