import numpy as np
import cmath
from numpy.typing import NDArray


def sqrt_complex(s):
    s = complex(s)
    modulus = abs(s)
    phase = cmath.phase(s)
    return np.sqrt(modulus) * complex(np.cos(phase / 2), np.sin(phase / 2))

def inv_matrix(A: NDArray):
    assert all([it == 2 for it in A.shape])
    a1, a2, a3, a4 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    determinant = a1 * a4 - a3 * a2
    assert not np.isclose(determinant, 0, atol=1e-5)
    return (1 / determinant) * np.array([[a4, -a2], [-a3, a1]])

def solve_lin_eq(a1, a2, a3, a4, b1, b2):
    # | a1 a2 | | c1 | = | b1 |
    # | a3 a4 | | c2 | = | b2 |
    A_inv = inv_matrix(np.array([[a1, a2], [a3, a4]]))
    return A_inv @ np.array([b1, b2])
