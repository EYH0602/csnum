import unittest
import numpy as np
from returns.result import Result, Success, Failure

from numerical_methods.eigen_system import power_method, inverse_power_method


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
        
    def test_inverse_power_method(self):
        A = np.array([
            [-4, 14, 0],
            [-5, 13, 0],
            [-1, 0, 2]
        ])
        x0 = np.array([[1, 1, 1]]).T
        mu, x = inverse_power_method(A, x0).unwrap()
        true_x = np.array([[1, 0.7142858, -0.2499996]]).T
        self.assertTrue(np.isclose(6.0000017, mu, atol=1e-2))
        self.assertTrue(np.allclose(x, true_x, atol=1e-2))



if __name__ == "__main__":
    unittest.main()
