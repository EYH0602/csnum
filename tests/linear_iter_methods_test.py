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

    def test_sor_conv(self):
        # see textbook example 1 on P.464
        # for the appropriate value of w
        # sor should be faster than gauss_seidel by a order of 2

        A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
        b = np.array([[24, 30, -24]]).T
        x0 = np.zeros(shape=b.shape)
        
        _, sor_it = sor(A, b, x0, w=1.25, max_iter=100, return_iter=True)
        _, gs_it = gauss_seidel(A, b, x0, max_iter=100, return_iter=True)
        
        self.assertTrue(gs_it / sor_it >= 2)


if __name__ == "__main__":
    unittest.main()
