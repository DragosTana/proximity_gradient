import numpy as np
import numba as nb

def _check_solver(solver, penalty, dual):

    return solver

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@nb.jit(nopython=True, parallel=True)
def fast_matmul(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Inner matrix dimensions must agree."

    C = np.zeros((n, p), dtype=A.dtype)

    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]

    return C
