import unittest
import numpy as np

from numerical_methods.linear_direct_methods import (
    gaussian_elimination,
    lu_factorization,
    back_substitution,
    lu_solve,
    gauss_solve,
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

        l, u = lu_factorization(A).unwrap()
        a, m = gaussian_elimination(A).unwrap()

        self.assertTrue(np.all(np.equal(l, a)))
        self.assertTrue(np.all(np.equal(u, m)))

    def test_back_substitution(self):
        A = np.matrix([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
        b = np.array([3, 2, 1]).reshape((3, 1))

        mat, _ = gaussian_elimination(np.hstack((A, b))).unwrap()
        self.assertTrue(
            np.allclose(
                back_substitution(mat[:, :-1], mat[:, [-1]]), np.linalg.solve(A, b)
            )
        )

    def test_gauss_solve(self):
        n = 5
        A = 5 * np.eye(n) + np.random.normal(size=(n, n))
        b = np.random.normal(size=(n, 1))

        def test(p):
            return np.allclose(
                gauss_solve(A, b, pivot=p).unwrap(), np.linalg.solve(A, b)
            )

        self.assertTrue(test("none"))
        self.assertTrue(test("partial"))
        self.assertTrue(test("scaled_partial"))

    def test_lu_solve(self):
        A = np.array([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
        b = np.array([3, 2, 1]).reshape((3, 1))

        u, l = lu_factorization(A).unwrap()
        self.assertTrue(np.allclose(lu_solve(l, u, b), np.linalg.solve(A, b)))


if __name__ == "__main__":
    unittest.main()
