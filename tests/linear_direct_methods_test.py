import unittest
import numpy as np

from numerical_methods.linear_direct_methods import (
    gaussian_elimination,
    lu_decomposition,
)


class TestRoot1Var(unittest.TestCase):
    def test_gaussian_elimination_none(self):
        A = np.array([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
        a, m = gaussian_elimination(A).unwrap()

        self.assertTrue(np.allclose(a, np.triu(a)))  # check if upper triangular
        self.assertTrue(np.allclose(m, np.tril(m)))  # check if lower triangular

    def test_gaussian_elimination_partial(self):
        A = np.array([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
        a, m = gaussian_elimination(A, pivot="partial").unwrap()

        self.assertTrue(np.allclose(a, np.triu(a)))  # check if upper triangular
        self.assertTrue(np.allclose(m, np.tril(m)))  # check if lower triangular

    def test_lu_decomposition(self):
        A = np.array([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)

        l, u = lu_decomposition(A).unwrap()
        a, m = gaussian_elimination(A).unwrap()

        self.assertTrue(np.all(np.equal(l, a)))
        self.assertTrue(np.all(np.equal(u, m)))


if __name__ == "__main__":
    unittest.main()
