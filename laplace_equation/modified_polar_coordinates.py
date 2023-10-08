"""
In this version we have modified the radius coordinate so that it distorts depending on how close we are to the surface
of the cylinder. The new independent variable is denoted 'rho' (not to be confused with the density) and the formula for
this transformation is given below:

r = r0 * e ^ (alpha * rho)

where 'alpha' is some constant, 'r0' is the radius of the cylinder, and 'e' is Euler's constant.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def dr_from_drho(r0, alpha, rho, drho):
    r = r0 * np.exp(alpha * rho)
    return alpha * r * drho


def boundary_conditions(grid, u, rf, da):
    m, n = grid.shape
    for j in range(n):
        theta = j * da
        grid[m - 1, j] = u * rf * np.cos(theta)
    return grid


def xy_location(m_rows, n_columns, r0, alpha, drho, da):
    X, Y = np.zeros((m_rows, n_columns)), np.zeros((m_rows, n_columns))
    for i in range(m_rows):
        r = r0 * np.exp(alpha * i * drho)
        for j in range(n_columns):
            theta = j * da
            X[i, j] = r * np.cos(theta)
            Y[i, j] = r * np.sin(theta)
    return X, Y


# method that performs the 'relaxation' technique to solve Laplace's equation in rectangular coordinates
def relaxation(grid, alpha, drho, da):
    m, n = grid.shape
    for i in range(m - 1):
        for j in range(n):
            # here we must apply the approximation for the first derivative in the radius direction
            # d(phi)/dr = 0 so that continuity is preserved
            if i == 0:
                # grid[i, j] = (4 * grid[i + 1, j] - grid[i + 2, j]) / 3  # 2nd order
                grid[i, j] = (18 * grid[i + 1, j] - 9 * grid[i + 2, j] + 2 * grid[i + 3, j]) / 11  # 3rd order
                continue
            cte1 = 1 / (alpha ** 2 * drho ** 2)
            cte2 = 1 / (da ** 2)
            cte0 = 2 * (cte1 + cte2)
            if j == 0:
                grid[i, j] = 1 / cte0 * (cte1 * (grid[i+1, j] + grid[i-1, j]) +
                                         cte2 * (2 * grid[i, j+1]))
            elif j == n - 1:
                grid[i, j] = 1 / cte0 * (cte1 * (grid[i+1, j] + grid[i-1, j]) +
                                         cte2 * (2 * grid[i, j-1]))
            else:
                grid[i, j] = 1 / cte0 * (cte1 * (grid[i+1, j] + grid[i-1, j]) +
                                         cte2 * (grid[i, j+1] + grid[i, j-1]))

    return grid


def gradient_grid(grid, r0, alpha, drho, da):
    m, n = grid.shape
    grad_r, grad_theta = np.zeros((m, n)), np.zeros((m, n))
    for i in range(m - 1):
        r = r0 * np.exp(alpha * i * drho)
        for j in range(n):
            grad_r[i, j] = 1 / (alpha * r) * (grid[i + 1, j] - grid[i - 1, j]) / (2 * drho)
            if j == 0:
                grad_theta[i, j] = 0
            elif j == n - 1:
                grad_theta[i, j] = 0
            else:
                grad_theta[i, j] = (grid[i, j + 1] - grid[i, j - 1]) / (2 * da)
    grad_r[m - 1, :] = grad_r[m - 2, :]
    grad_theta[m - 1, :] = grad_theta[m - 2, :]
    return grad_r, grad_theta


def gradient_grid_cartesian(grad_r, grad_theta, r0, alpha, drho, da):
    m, n = grad_r.shape
    grad_x, grad_y = np.zeros((m, n)), np.zeros((m, n))
    for i in range(m - 1):
        r = r0 * np.exp(alpha * i * drho)
        for j in range(n):
            theta = j * da
            grad_x[i, j] = np.cos(theta) * grad_r[i, j] - 1 / r * np.sin(theta) * grad_theta[i, j]
            grad_y[i, j] = np.sin(theta) * grad_r[i, j] + 1 / r * np.cos(theta) * grad_theta[i, j]
    grad_x[m - 1, :] = grad_x[m - 2, :]
    grad_y[m - 1, :] = grad_y[m - 2, :]
    return grad_x, grad_y


def theoretical_solution(grid, u0, r0, alpha, drho, da):
    m, n = grid.shape
    solution = np.zeros((m, n))
    for i in range(m):
        r = r0 * np.exp(alpha * i * drho)
        for j in range(n):
            theta = j * da
            solution[i, j] = u0 * r * (1 + r0 ** 2 / r ** 2) * np.cos(theta)
    return solution


def pressure_field(grad_x, grad_y, p_infinity, rho, u_infinity):
    m, n = grad_x.shape
    speed = (grad_x ** 2 + grad_y ** 2) ** 0.5
    pressure = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            pressure[i, j] = 0.5 * rho * (u_infinity ** 2 - speed[i, j] ** 2) + p_infinity
    return pressure


def pressure_coefficient(p, p_infinity, u_infinity, rho):
    m, n = p.shape
    cp = np.zeros(p.shape)
    for i in range(m):
        for j in range(n):
            cp[i, j] = (p[i, j] - p_infinity) / (0.5 * rho * u_infinity ** 2)
    return cp


if __name__ == '__main__':

    # Initial parameters
    input_alpha = 0.1
    r_size, theta_size = 250, 250
    u_speed = 10
    input_r0, input_rf = 1, 10
    input_rho, input_rhof = 0, 1 / input_alpha * np.log(input_rf / input_r0)
    input_drho = input_rhof / (r_size - 1)
    input_p0, input_density = 0, 1
    input_da, n_iterations = np.pi / (theta_size - 1), 50_000

    # Initialize solution with boundary conditions
    phi = np.zeros((r_size, theta_size))
    phi = boundary_conditions(phi, u_speed, input_rf, input_da)

    # Run multiple iterations for final solution
    original_time = time.time()
    time_now = original_time
    for iteration in range(1, n_iterations + 1):
        phi = relaxation(phi, input_alpha, input_drho, input_da)
        if iteration % 1000 == 0:
            print(f"Iteration {iteration} took: {time.time() - time_now:.2f} seconds")
            time_now = time.time()
    print(f"\nTotal time taken: {time.time() - original_time:.2f} seconds\n")

    # Save the numpy grid with a specific format
    np.savetxt("phi_grid.csv", phi, delimiter=",", fmt="%10.5f")

    # Get rectangular coordinates from polar coordinates and respective gradients
    x, y = xy_location(r_size, theta_size, input_r0, input_alpha, input_drho, input_da)
    grad_r, grad_theta = gradient_grid(phi, input_r0, input_alpha, input_drho, input_da)
    grad_x, grad_y = gradient_grid_cartesian(grad_r, grad_theta, input_r0, input_alpha, input_drho, input_da)
    p = pressure_field(grad_x, grad_y, input_p0, input_density, u_speed)

    # Get theoretical solution
    phi_theory = theoretical_solution(phi, u_speed, input_r0, input_alpha, input_drho, input_da)
    grad_r_theory, grad_theta_theory = gradient_grid(phi_theory, input_r0, input_alpha, input_drho, input_da)
    grad_x_theory, grad_y_theory = gradient_grid_cartesian(grad_r_theory, grad_theta_theory, input_r0, input_alpha, input_drho, input_da)
    p_theory = pressure_field(grad_x_theory, grad_y_theory, input_p0, input_density, u_speed)

    # Fix the size of 'x', 'y', 'grad_x', 'grad_y', 'pressure' before plotting
    x = x[1:, :]
    y = y[1:, :]
    grad_x = grad_x[1:, :]
    grad_y = grad_y[1:, :]
    p = p[1:, :]
    grad_x_theory = grad_x_theory[1:, :]
    grad_y_theory = grad_y_theory[1:, :]
    p_theory = p_theory[1:, :]

    # Calculate the absolute speed at each point
    speed = (grad_x ** 2 + grad_y ** 2) ** 0.5
    speed_theory = (grad_x_theory ** 2 + grad_y_theory ** 2) ** 0.5

    # Calculate the pressure coefficient at the surface of the disk
    cp = pressure_coefficient(p, input_p0, u_speed, input_density)
    cp_surface = cp[0, :]
    cp_theory = pressure_coefficient(p_theory, input_p0, u_speed, input_density)
    cp_surface_theory = cp_theory[0, :]

    # Total error in speed calculation across the numerical grid
    eps = 1e-5
    error_matrix = 100 * np.abs(speed - speed_theory) / (speed_theory + eps * np.random.random(size=speed_theory.shape))
    error = np.average(error_matrix)
    print(f"Average error: {error:.1f}%")

    # Make plot
    plt.quiver(x, y, grad_x, grad_y, speed)  # numerical
    # plt.quiver(x, y, grad_x_theory, grad_y_theory, speed_theory)  # theory
    # plt.quiver(x, y, grad_x, grad_y, p)  # pressure field numerical
    # plt.quiver(x, y, grad_x_theory, grad_y_theory, p_theory)  # pressure field theory
    plt.show()

    plt.plot(cp_surface, 'red')
    plt.plot(cp_surface_theory, 'blue')
    plt.show()
