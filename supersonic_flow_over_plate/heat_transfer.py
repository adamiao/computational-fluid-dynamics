"""
This script has the purpose of defining all the heat transfer functions that are used in MacCormack's method.
Here we are interested in having both the backward and forward versions of the heat transfers that are used in the
derived variables E and F.
"""

import numpy as np
from supersonic_flow_over_plate.physics import compute_thermal_conductivity
from supersonic_flow_over_plate.constants import (
    N_X,
    N_Y,
    DELTA_X,
    DELTA_Y,
)


def qx_E_backward(T):
    qx = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            if j == 0:
                T_derivative = (T[i, j + 1] - T[i, j]) / DELTA_X
            else:
                T_derivative = (T[i, j] - T[i, j - 1]) / DELTA_X
            k = compute_thermal_conductivity(T[i, j])
            qx[i, j] = -k * T_derivative
    return qx


def qx_E_forward(T):
    qx = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            if j == N_X - 1:
                T_derivative = (T[i, j] - T[i, j - 1]) / DELTA_X
            else:
                T_derivative = (T[i, j + 1] - T[i, j]) / DELTA_X
            k = compute_thermal_conductivity(T[i, j])
            qx[i, j] = -k * T_derivative
    return qx


def qy_F_backward(T):
    qy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            if i == N_Y - 1:
                T_derivative = (T[i - 1, j] - T[i, j]) / DELTA_Y
            else:
                T_derivative = (T[i, j] - T[i + 1, j]) / DELTA_Y
            k = compute_thermal_conductivity(T[i, j])
            qy[i, j] = -k * T_derivative
    return qy


def qy_F_forward(T):
    qy = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        for j in range(N_X):
            if i == 0:
                T_derivative = (T[i, j] - T[i + 1, j]) / DELTA_Y
            else:
                T_derivative = (T[i - 1, j] - T[i, j]) / DELTA_Y
            k = compute_thermal_conductivity(T[i, j])
            qy[i, j] = -k * T_derivative
    return qy
