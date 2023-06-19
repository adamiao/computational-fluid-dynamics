"""
This script has the purpose of defining all the shear stress functions that are used in MacCormack's method.
Here we are interested in having both the backward and forward versions of the shear stress that are used in the
derived variables E and F.
"""

import numpy as np
from supersonic_flow_over_plate.physics import compute_viscosity
from supersonic_flow_over_plate.constants import (
    N_X,
    N_Y,
    DELTA_X,
    DELTA_Y,
)


def tau_xy_E_backward(u, v, T):
    tau_xy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the y-speed derivative
            if j == 0:
                v_derivative = (v[i, j + 1] - v[i, j]) / DELTA_X
            else:
                v_derivative = (v[i, j] - v[i, j - 1]) / DELTA_X

            # Calculate the x-speed derivative
            if i == 0:
                u_derivative = (u[i, j] - u[i + 1, j]) / DELTA_Y
            elif i == N_Y - 1:
                u_derivative = (u[i - 1, j] - u[i, j]) / DELTA_Y
            else:
                u_derivative = (u[i - 1, j] - u[i + 1, j]) / (2 * DELTA_Y)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xy
            tau_xy[i, j] = mu * (u_derivative + v_derivative)
    return tau_xy


def tau_xy_E_forward(u, v, T):
    tau_xy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the y-speed derivative
            if j == N_X - 1:
                v_derivative = (v[i, j] - v[i, j - 1]) / DELTA_X
            else:
                v_derivative = (v[i, j + 1] - v[i, j]) / DELTA_X

            # Calculate the x-speed derivative
            if i == 0:
                u_derivative = (u[i, j] - u[i + 1, j]) / DELTA_Y
            elif i == N_Y - 1:
                u_derivative = (u[i - 1, j] - u[i, j]) / DELTA_Y
            else:
                u_derivative = (u[i - 1, j] - u[i + 1, j]) / (2 * DELTA_Y)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xy
            tau_xy[i, j] = mu * (u_derivative + v_derivative)
    return tau_xy


def tau_xy_F_backward(u, v, T):
    tau_xy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the x-speed derivative
            if i == N_Y - 1:
                u_derivative = (u[i - 1, j] - u[i, j]) / DELTA_Y
            else:
                u_derivative = (u[i, j] - u[i + 1, j]) / DELTA_Y

            # Calculate the y-speed derivative
            if j == 0:
                v_derivative = (v[i, j + 1] - v[i, j]) / DELTA_X
            elif j == N_X - 1:
                v_derivative = (v[i, j] - v[i, j - 1]) / DELTA_X
            else:
                v_derivative = (v[i, j + 1] - v[i, j - 1]) / (2 * DELTA_X)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xy
            tau_xy[i, j] = mu * (u_derivative + v_derivative)
    return tau_xy


def tau_xy_F_forward(u, v, T):
    tau_xy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the x-speed derivative
            if i == 0:
                u_derivative = (u[i, j] - u[i + 1, j]) / DELTA_Y
            else:
                u_derivative = (u[i - 1, j] - u[i, j]) / DELTA_Y

            # Calculate the y-speed derivative
            if j == 0:
                v_derivative = (v[i, j + 1] - v[i, j]) / DELTA_X
            elif j == N_X - 1:
                v_derivative = (v[i, j] - v[i, j - 1]) / DELTA_X
            else:
                v_derivative = (v[i, j + 1] - v[i, j - 1]) / (2 * DELTA_X)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xy
            tau_xy[i, j] = mu * (u_derivative + v_derivative)
    return tau_xy


def tau_xx_E_backward(u, v, T):
    tau_xx = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the x-speed derivative
            if j == 0:
                u_derivative = (u[i, j + 1] - u[i, j]) / DELTA_X
            else:
                u_derivative = (u[i, j] - u[i, j - 1]) / DELTA_X

            # Calculate the y-speed derivative
            if i == 0:
                v_derivative = (v[i, j] - v[i + 1, j]) / DELTA_Y
            elif i == N_Y - 1:
                v_derivative = (v[i - 1, j] - v[i, j]) / DELTA_Y
            else:
                v_derivative = (v[i - 1, j] - v[i + 1, j]) / (2 * DELTA_Y)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xx ( lambda = -2/3 mu )
            tau_xx[i, j] = 4 / 3 * mu * u_derivative - 2 / 3 * mu * v_derivative
    return tau_xx


def tau_xx_E_forward(u, v, T):
    tau_xx = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the x-speed derivative
            if j == N_X - 1:
                u_derivative = (u[i, j] - u[i, j - 1]) / DELTA_X
            else:
                u_derivative = (u[i, j + 1] - u[i, j]) / DELTA_X

            # Calculate the y-speed derivative
            if i == 0:
                v_derivative = (v[i, j] - v[i + 1, j]) / DELTA_Y
            elif i == N_Y - 1:
                v_derivative = (v[i - 1, j] - v[i, j]) / DELTA_Y
            else:
                v_derivative = (v[i - 1, j] - v[i + 1, j]) / (2 * DELTA_Y)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_xx ( lambda = -2/3 mu )
            tau_xx[i, j] = 4 / 3 * mu * u_derivative - 2 / 3 * mu * v_derivative
    return tau_xx


def tau_yy_F_backward(u, v, T):
    tau_yy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the y-speed derivative
            if i == N_Y - 1:
                v_derivative = (v[i - 1, j] - v[i, j]) / DELTA_Y
            else:
                v_derivative = (v[i, j] - v[i + 1, j]) / DELTA_Y

            # Calculate the x-speed derivative
            if j == 0:
                u_derivative = (u[i, j + 1] - u[i, j]) / DELTA_X
            elif j == N_X - 1:
                u_derivative = (u[i, j] - u[i, j - 1]) / DELTA_X
            else:
                u_derivative = (u[i, j + 1] - u[i, j - 1]) / (2 * DELTA_X)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_yy ( lambda = -2/3 mu )
            tau_yy[i, j] = - 2 / 3 * mu * u_derivative + 4 / 3 * mu * v_derivative
    return tau_yy


def tau_yy_F_forward(u, v, T):
    tau_yy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            # Calculate the y-speed derivative
            if i == 0:
                v_derivative = (v[i, j] - v[i + 1, j]) / DELTA_Y
            else:
                v_derivative = (v[i - 1, j] - v[i, j]) / DELTA_Y

            # Calculate the x-speed derivative
            if j == 0:
                u_derivative = (u[i, j + 1] - u[i, j]) / DELTA_X
            elif j == N_X - 1:
                u_derivative = (u[i, j] - u[i, j - 1]) / DELTA_X
            else:
                u_derivative = (u[i, j + 1] - u[i, j - 1]) / (2 * DELTA_X)

            # Compute viscosity for node i, j
            mu = compute_viscosity(T[i, j])

            # Compute tau_yy ( lambda = -2/3 mu )
            tau_yy[i, j] = - 2 / 3 * mu * u_derivative + 4 / 3 * mu * v_derivative
    return tau_yy
