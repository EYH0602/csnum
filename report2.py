import numpy as np
from returns.maybe import Maybe, Some, Nothing
from numerical_methods.linear_direct_methods import (
    gaussian_elimination,
    back_substitution,
    lu_decomposition,
)

A = np.matrix([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
b = np.array([3, 2, 1])

L, U = lu_decomposition(A).unwrap()
print(L)
print(U)
print(gaussian_elimination(A, "partial").unwrap()[0])
