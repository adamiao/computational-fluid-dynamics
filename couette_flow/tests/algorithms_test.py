import numpy as np
from couette_flow.implicit_solution.algorithms import thomas, tridiagonal


def test_thomas():
    # Tridiagonal linear system: A * x = b
    a = np.array([(0.11, 0.52, 0, 0), (0.43, 0.5, 0.84, 0), (0, 0.23, 0.34, 0.67), (0, 0, 0.91, 0.98)])
    b = np.array([0.1, 0.2, 0.3, 0.4])

    # Run the Thomas' algorithm
    solution = thomas(a, b)

    # Compare the solutions
    assert sum(abs(solution - np.linalg.solve(a, b))) < 1e-6


def test_tridiagonal():
    # Test matrix in tridiagonal form
    true_tridiagonal_matrix = np.array([[2, 1, 0, 0], [-1, 2, 1, 0], [0, -1, 2, 1], [0, 0, -1, 2]])

    # Create the same tridiagonal matrix using the 'tridiagonal' function
    tridiagonal_matrix = tridiagonal(-1, 2, 1, 4)

    # Compare the solutions
    assert sum(sum(abs(true_tridiagonal_matrix - tridiagonal_matrix))) < 1e-6
