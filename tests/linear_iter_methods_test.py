import unittest
import numpy as np

from numerical_methods.linear_iter_methods import jacobi, gauss_seidel, sor


def generate_random(n):
    A = n / 2 * np.identity(n) + np.random.normal(size=(n, n))
    b = np.random.normal(size=(n, 1))
    return A, b


class TestRoot1Var(unittest.TestCase):
    def test_jacobi(self):
        for _ in range(10):
            A, b = generate_random(50)
            x0 = np.zeros(shape=b.shape)

            x = jacobi(A, b, x0, max_iter=20)
            self.assertTrue(np.allclose(np.linalg.solve(A, b), x))

    def test_gauss_seidel(self):
        for _ in range(10):
            A, b = generate_random(50)
            x0 = np.zeros(shape=b.shape)

            x = gauss_seidel(A, b, x0, max_iter=20)
            self.assertTrue(np.allclose(np.linalg.solve(A, b), x))

    def test_sor(self):
        for _ in range(10):
            A, b = generate_random(50)
            x0 = np.zeros(shape=b.shape)

            x = sor(A, b, x0, max_iter=20)
            self.assertTrue(np.allclose(np.linalg.solve(A, b), x))


if __name__ == "__main__":
    unittest.main()
