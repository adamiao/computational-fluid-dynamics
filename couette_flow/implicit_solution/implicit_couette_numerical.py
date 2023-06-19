import numpy as np
import matplotlib.pyplot as plt
from couette_flow.implicit_solution.constants import E, N, DELTA_Y
from couette_flow.implicit_solution.initial_and_boundary_conditions import initial_conditions
from couette_flow.implicit_solution.algorithms import thomas, tridiagonal


def compute_iteration(u):
    K = np.array([(1 - E) * u[j] + 0.5 * E * (u[j - 1] + u[j + 1]) for j in range(1, N - 1)])
    K[-1] += 0.5 * E * u[-1]  # notice the last row of the linear system shown in equation 9.32, on page 424
    matrix_a = tridiagonal(-E / 2, 1 + E, -E / 2, N - 2)
    u[1:-1] = thomas(matrix_a, K)
    return u


if __name__ == '__main__':
    # Initialize speed profile and nondimensional vertical distance
    speed = initial_conditions()
    y = [i * DELTA_Y for i in range(N)]

    # Run the solution for multiple iterations while saving specific solutions at desired times
    distinct_times_of_interest = [0, 2, 12, 36, 60, 240]
    solutions = [speed.copy()]  # start keeping track of the solutions
    for n in range(1, 241):
        speed = compute_iteration(speed)
        if n in distinct_times_of_interest:
            solutions.append(speed.copy())

    # Plot the solutions that were saved
    fig = plt.figure(figsize=(10, 8))
    for i, solution in enumerate(solutions):
        plt.plot(solution, y, label=f'{distinct_times_of_interest[i]}' + r'$\Delta t$')
        plt.scatter(solution, y)

    major_ticks = np.arange(0, 1.1, 0.1)
    minor_ticks = np.arange(0, 1.1, 0.05)
    plt.xlabel(r'$u/u_e$')
    plt.xticks(major_ticks)
    plt.xticks(minor_ticks, minor=True)
    plt.ylabel(r'$y/D$')
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Unsteady Couette Flow')

    plt.legend()
    # plt.savefig('couette_flow_unsteady_E_equal_1.png')
    plt.show()
