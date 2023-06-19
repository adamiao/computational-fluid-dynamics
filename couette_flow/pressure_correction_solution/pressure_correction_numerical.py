import numpy as np
import matplotlib.pyplot as plt
from couette_flow.pressure_correction_solution.initial_and_boundary_conditions import (
    initial_conditions,
    boundary_conditions,
)
from couette_flow.pressure_correction_solution.constants import (
    VISCOSITY, DENSITY,
    NX_P, NY_P, NX_U, NX_V, NY_V,
    DELTA_X, DELTA_Y, DELTA_T,
    ITERATIONS, RELAXATION_ITERATIONS,
    RELAXATION_CONSTANT,
)


def relaxation(p, u, v):
    a0, a1 = -DELTA_T / DELTA_X ** 2, -DELTA_T / DELTA_Y ** 2
    a = -2 * (a0 + a1)
    p_prime = np.zeros(shape=p.shape)
    for k in range(RELAXATION_ITERATIONS):
        for i in range(1, NY_P - 1):
            for j in range(1, NX_P - 1):
                b = DENSITY / DELTA_X * (u[i, j + 1] - u[i, j]) + DENSITY / DELTA_Y * (v[i, j + 1] - v[i + 1, j + 1])
                p_prime[i, j] = -a0 / a * (p_prime[i, j - 1] + p_prime[i, j + 1]) +\
                                -a1 / a * (p_prime[i - 1, j] + p_prime[i + 1, j]) +\
                                -b / a
    p += RELAXATION_CONSTANT * p_prime


def update_u(i, j, p, u, v):

    # Helper variables
    v_bar = 0.5 * (v[i, j + 1] + v[i, j + 2])
    v_dbar = 0.5 * (v[i + 1, j + 1] + v[i + 1, j + 2])

    A0 = -(
            (u[i, j + 2] ** 2 - u[i, j] ** 2) / (2 * DELTA_X) +
            (u[i - 1, j + 1] * v_bar - u[i + 1, j + 1] * v_dbar) / (2 * DELTA_Y)
    )
    A1 = (u[i, j + 2] - 2 * u[i, j + 1] + u[i, j]) / DELTA_X ** 2 +\
         (u[i - 1, j + 1] - 2 * u[i, j + 1] + u[i + 1, j + 1]) / DELTA_Y ** 2
    A = A0 + (VISCOSITY / DENSITY) * A1

    u[i, j + 1] = u[i, j + 1] + A * DELTA_T - DELTA_T / (DENSITY * DELTA_X) * (p[i, j + 1] - p[i, j])

    return


def update_v(i, j, p, u, v):

    # Helper variables
    u_bar = 0.5 * (u[i - 1, j + 1] + u[i, j + 1])
    u_dbar = 0.5 * (u[i - 1, j] + u[i, j])

    B0 = -(
            (v[i, j + 2] * u_bar - v[i, j] * u_dbar) / (2 * DELTA_X) +
            (v[i - 1, j + 1] ** 2 - v[i + 1, j + 1] ** 2) / (2 * DELTA_Y)
    )
    B1 = (v[i, j + 2] - 2 * v[i, j + 1] + v[i, j]) / DELTA_X ** 2 +\
         (v[i - 1, j + 1] - 2 * v[i, j + 1] + v[i + 1, j + 1]) / DELTA_Y ** 2
    B = B0 + (VISCOSITY / DENSITY) * B1

    v[i, j + 1] = v[i, j + 1] + B * DELTA_T - DELTA_T / (DENSITY * DELTA_Y) * (p[i - 1, j] - p[i, j])

    return


def iteration(p, u, v):
    # Compute the pressure correction from the current speed profile
    relaxation(p, u, v)
    for i in range(1, NY_P - 1):
        for j in range(NX_P - 1):
            update_u(i, j, p, u, v)
            update_v(i, j, p, u, v)
    boundary_conditions(p, u, v)


if __name__ == '__main__':
    # Initialize flow variables
    pressure, u_speed, v_speed = initial_conditions()

    # Store some of the solutions to later check the computed flow speeds
    y_solutions, x_solutions, y_iterations, x_iterations = [], [], [0, 1, 4, 50], [4, 20, 50, 150, 300]
    for n in range(ITERATIONS):
        if n in y_iterations:
            y_solutions.append(v_speed[::-1, NX_V // 2].copy())
        if n in x_iterations:
            x_solutions.append(u_speed[::-1, NX_U // 2].copy())
        iteration(pressure, u_speed, v_speed)

    # Variables that will be used for plotting
    height = np.array([i * DELTA_Y for i in range(NY_V)])

    # Plot the solutions that were saved
    fig = plt.figure(figsize=(10, 8))
    for n, y_solution in enumerate(y_solutions):
        plt.plot(y_solution, height, label=f'K={y_iterations[n]}')
        plt.scatter(y_solution, height)

    x_major_ticks = np.arange(0, 0.5, 0.05)
    x_minor_ticks = np.arange(0, 0.5, 0.05)
    y_major_ticks = np.arange(0, 0.011, 0.002)
    y_minor_ticks = np.arange(0, 0.011, 0.001)
    plt.xlabel(r'v ' + r'$(ft/s)$')
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.ylabel(r'y ' + r'$(ft)$')
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Unsteady Couette Flow - Pressure Correction')

    plt.legend()
    # plt.savefig('couette_flow_pressure_correction_y_speed.png')
    plt.show()

    # Plot the x-coordinate speed profile
    fig = plt.figure(figsize=(10, 8))
    for n, x_solution in enumerate(x_solutions):
        plt.plot(x_solution, height, label=f'K={x_iterations[n]}')
        plt.scatter(x_solution, height)

    x_major_ticks = np.arange(0, 1.1, 0.1)
    x_minor_ticks = np.arange(0, 1.1, 0.05)
    y_major_ticks = np.arange(0, 0.011, 0.002)
    y_minor_ticks = np.arange(0, 0.011, 0.001)
    plt.xlabel(r'u ' + r'$(ft/s)$')
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.ylabel(r'y ' + r'$(ft)$')
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Unsteady Couette Flow - Pressure Correction')

    plt.legend()
    # plt.savefig('couette_flow_pressure_correction_x_speed.png')
    plt.show()
