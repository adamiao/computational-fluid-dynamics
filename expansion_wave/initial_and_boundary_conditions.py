import numpy as np
from expansion_wave.physics import (
    prandtl_meyer_function,
    inverse_prandtl_meyer_function,
    pressure_ratio,
    temperature_ratio,
    density,
)
from expansion_wave.variable_conversion import decode_primitive_variables
from expansion_wave.constants import (
    GAMMA,
    R,
    N_X,
    N_Y,
    MACH_0,
    P_0,
    RHO_0,
    T_0,
    X_EXPANSION,
    THETA,
)


def initial_conditions():
    # Calculate the initial velocity
    u, v = MACH_0 * (GAMMA * R * T_0) ** 0.5, 0.0

    # Create the grid for the derived variables 'F1', 'F2', 'F3', 'F4'
    F1, F2, F3, F4 = np.zeros((N_Y, N_X)), np.zeros((N_Y, N_X)), np.zeros((N_Y, N_X)), np.zeros((N_Y, N_X))
    F1[:, 0] = RHO_0 * u
    F2[:, 0] = RHO_0 * u ** 2 + P_0
    F3[:, 0] = RHO_0 * u * v
    F4[:, 0] = GAMMA / (GAMMA - 1) * P_0 * u + RHO_0 * u * ((u ** 2 + v ** 2) / 2)

    # Create a one dimensional numpy array to keep track of the distance at each node location
    X = np.zeros(N_X)

    return F1, F2, F3, F4, X


def boundary_conditions(F1, F2, F3, F4, xi):

    # Calculate primitive variables from derived variables
    rho, u, v, p, T = decode_primitive_variables(F1, F2, F3, F4)
    a = (GAMMA * R * T[0]) ** 0.5

    # Check geometry to see how we calculate flow angle with the wall
    if xi <= X_EXPANSION:
        phi = np.arctan(v[0] / u[0])
    else:
        phi = np.radians(THETA) - np.arctan(abs(v[0] / u[0]))

    # If absolute angle is small don't do anything
    if abs(phi) < 1e-3:
        return

    # Determine the calculated values of the Mach number and the Prandtl-Meyer function
    M_cal = (u[0] ** 2 + v[0] ** 2) ** 0.5 / a
    f_cal = prandtl_meyer_function(M_cal)

    # Calculate the actual values of the Prandtl-Meyer function and the Mach number
    f_act = f_cal + phi
    M_act = inverse_prandtl_meyer_function(f_act)

    # Update the values of pressure, temperature, and density
    p_act = p[0] * pressure_ratio(M_cal, M_act)
    T_act = T[0] * temperature_ratio(M_cal, M_act)
    rho_act = density(p_act, T_act)

    # Now that we have updated the values of the primitive variables, we must ensure that the speed perpendicular to the
    # wall is in fact giving us a velocity parallel to the wall. We do that by only changing the 'v' speed.
    if xi <= X_EXPANSION:
        v[0] = 0.0
    else:
        v[0] = -u[0] * np.tan(np.radians(THETA))

    # Update the derived variables based on these new values of pressure, temperature, and density
    F1[0] = rho_act * u[0]
    F2[0] = rho_act * u[0] ** 2 + p_act
    F3[0] = rho_act * u[0] * v[0]
    F4[0] = GAMMA / (GAMMA - 1) * p_act * u[0] + rho_act * u[0] * ((u[0] ** 2 + v[0] ** 2) / 2)

    return
