import unittest
import numpy as np
from returns.result import Result, Success, Failure

from csnum.eigen_system import (
    power_method,
    inverse_power_method,
    wielandt_deflation,
)


def rand_mat(n=50):
    B = np.random.normal(size=(n, n))
    A = np.eye(n) / 2 + B + B.T
    return A


class TestRoot1Var(unittest.TestCase):
    def test_power_method(self):
        A = np.array([[-2, -3], [6, 7]])
        x0 = np.array([[1, 1]]).T
        mu, x = power_method(A, x0).unwrap()
        true_x = np.array([[-0.49908, 1]]).T
        self.assertTrue(np.isclose(4.002, mu, atol=1e-2))
        self.assertTrue(np.allclose(x, true_x, atol=1e-2))

        A = np.array([[4, -1, 1], [-1, 3, -2], [1, -2, 3]])
        x0 = np.array([[1, 0, 0]]).T
        mu, x = power_method(A, x0, max_iter=20).unwrap()
        true_x = np.array([[1, -0.997076, 0.997076]]).T
        self.assertTrue(np.isclose(6.000184, mu, atol=1e-3))
        self.assertTrue(np.allclose(x, true_x, atol=1e-2))

    def test_power_method_random(self):
        for _ in range(100):
            n = 50
            A = rand_mat(n=n)
            x0 = np.ones(shape=(n, 1))
            match power_method(A, x0, max_iter=100000000, thresh=1e-8):
                case Success((mu, _)):
                    xs, _ = np.linalg.eig(A)
                    x = xs[np.argmax(np.abs(xs))]
                    self.assertTrue(np.isclose(mu, x))
                case Failure(_):
                    self.fail()

    def test_inverse_power_method(self):
        A = np.array([[-4, 14, 0], [-5, 13, 0], [-1, 0, 2]])
        x0 = np.array([[1, 1, 1]]).T
        q = x0.T @ A @ x0 / (x0.T @ x0)
        mu, x = inverse_power_method(A, x0, max_iter=1000, thresh=1e-8, q=q).unwrap()
        true_x = np.array([[1, 0.7142858, -0.2499996]]).T
        self.assertTrue(np.isclose(6.0000017, mu, atol=1e-2))
        self.assertTrue(np.allclose(x, true_x, atol=1e-2))

    def test_inverse_power_method_random(self):
        for _ in range(100):
            n = 50
            A = rand_mat(n=n)
            x0 = np.ones(shape=(n, 1))
            match inverse_power_method(A, x0, max_iter=10000, thresh=1e-8):
                case Success((mu, _)):
                    xs, _ = np.linalg.eig(A)
                    x = xs[np.argmin(np.abs(xs))]
                    self.assertTrue(np.isclose(mu, x))
                case Failure(_):
                    self.fail()

    def test_wielandt_deflation(self):
        A = np.array([[4, -1, 1], [-1, 3, -2], [1, -2, 3]])
        x0 = np.array([[1, 0]]).T

        l = 6
        v = np.array([[1, -1, 1]]).T

        l_true = 3
        v_true = np.array([[-2, -1, 1]]).T
        l2, v2 = wielandt_deflation(
            A, x0, l, v, solver=power_method, thresh=1e-6
        ).unwrap()
        self.assertTrue(np.isclose(l2, l_true, rtol=1e-4))
        self.assertTrue(np.allclose(v2, v_true, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
