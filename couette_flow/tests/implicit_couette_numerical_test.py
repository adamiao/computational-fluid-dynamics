import numpy as np
from couette_flow.implicit_solution.initial_and_boundary_conditions import initial_conditions
from couette_flow.implicit_solution.implicit_couette_numerical import compute_iteration


def test_compute_iteration():
    solution_after_1_iteration = np.array([0.00000000e+00, 2.52156210e-11, 1.00862484e-10, 3.78234315e-10,
                                           1.41207478e-09, 5.27006479e-09, 1.96681844e-08, 7.34026727e-08,
                                           2.73942506e-07, 1.02236735e-06, 3.81552691e-06, 1.42397403e-05,
                                           5.31434342e-05, 1.98333996e-04, 7.40192551e-04, 2.76243621e-03,
                                           1.03095523e-02, 3.84757729e-02, 1.43593539e-01, 5.35898385e-01,
                                           1.00000000e+00])

    # Initialize problem
    speed = initial_conditions()

    # Run for 1 iteration
    speed = compute_iteration(speed)

    # Compare the solutions
    assert sum(abs(solution_after_1_iteration - speed)) < 1e-6
