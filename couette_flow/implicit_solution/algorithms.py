import numpy as np


def thomas(a, b):
    mod_a, mod_b = a.copy(), b.copy()

    # Sweep the matrix and update the values of the diagonal and the right side of the equation
    for i, row in enumerate(a[1:], 1):
        mod_a[i, i] = mod_a[i, i] - (mod_a[i, i - 1] * mod_a[i - 1, i]) / mod_a[i - 1, i - 1]
        mod_b[i] = mod_b[i] - (mod_a[i, i - 1] * mod_b[i - 1]) / mod_a[i - 1, i - 1]
        mod_a[i, i - 1] = 0

    # Compute the solution to the tridiagonal linear system
    solution = np.zeros(shape=b.shape)
    solution[-1] = mod_b[-1] / mod_a[-1, -1]
    for i, row in enumerate(mod_a[-2::-1], 2):
        solution[-i] = (mod_b[-i] - mod_a[-i, -i + 1] * solution[-i + 1]) / mod_a[-i, -i]

    return solution


def tridiagonal(L, D, U, N):
    matrix = np.diag([L for _ in range(N - 1)], -1) +\
             np.diag([D for _ in range(N)], 0) +\
             np.diag([U for _ in range(N - 1)], 1)
    return matrix
