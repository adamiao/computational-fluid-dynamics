"""
This script contains the heart of the algorithm: MacCormack's method.
"""

import numpy as np
from supersonic_flow_over_plate.boundary_conditions import boundary_conditions
from supersonic_flow_over_plate.constants import (
    N_X,
    N_Y,
    DELTA_X,
    DELTA_Y,
)
from supersonic_flow_over_plate.variables import (
    primitive_variables,
    compute_U,
    compute_E,
    compute_F,
    time_step,
)
from supersonic_flow_over_plate.heat_transfer import (
    qx_E_backward, qx_E_forward,
    qy_F_backward, qy_F_forward,
)
from supersonic_flow_over_plate.shear_stress import (
    tau_xy_E_backward, tau_xy_E_forward,
    tau_xy_F_backward, tau_xy_F_forward,
    tau_xx_E_backward, tau_xx_E_forward,
    tau_yy_F_backward, tau_yy_F_forward,
)


def maccormack(u, v, T, rho, p, boundary_type='constant_temperature'):
    # Compute the derived variables: U1, U2, U3, U5
    U1, U2, U3, U5 = compute_U(u, v, T, rho)
    E_total_in = U5.copy()

    # Determine the time step for the current iteration
    dt = time_step(u, v, T, rho)

    """
    PREDICTOR STEP
    """

    # Compute the other derived variables (the fact that we're in the predictor step dictates how the following
    # functions are called)
    tau_xx_E = tau_xx_E_backward(u, v, T)
    tau_xy_E = tau_xy_E_backward(u, v, T)
    tau_xy_F = tau_xy_F_backward(u, v, T)
    tau_yy_F = tau_yy_F_backward(u, v, T)
    qx_E = qx_E_backward(T)
    qy_F = qy_F_backward(T)
    E1, E2, E3, E5 = compute_E(u, v, rho, p, E_total_in, tau_xx_E, tau_xy_E, qx_E)
    F1, F2, F3, F5 = compute_F(u, v, rho, p, E_total_in, tau_yy_F, tau_xy_F, qy_F)

    dU1dt_predictor = np.zeros(shape=(N_Y, N_X))
    dU2dt_predictor = np.zeros(shape=(N_Y, N_X))
    dU3dt_predictor = np.zeros(shape=(N_Y, N_X))
    dU5dt_predictor = np.zeros(shape=(N_Y, N_X))
    for i in range(1, N_Y - 1):
        for j in range(1, N_X - 1):
            dU1dt_predictor[i, j] = -(E1[i, j + 1] - E1[i, j]) / DELTA_X - (F1[i - 1, j] - F1[i, j]) / DELTA_Y
            dU2dt_predictor[i, j] = -(E2[i, j + 1] - E2[i, j]) / DELTA_X - (F2[i - 1, j] - F2[i, j]) / DELTA_Y
            dU3dt_predictor[i, j] = -(E3[i, j + 1] - E3[i, j]) / DELTA_X - (F3[i - 1, j] - F3[i, j]) / DELTA_Y
            dU5dt_predictor[i, j] = -(E5[i, j + 1] - E5[i, j]) / DELTA_X - (F5[i - 1, j] - F5[i, j]) / DELTA_Y

    # Use the current estimate of the derivative for U1, U2, U3, U5 to determine the primitive predicted flow variables
    U1_pred = U1 + dU1dt_predictor * dt
    U2_pred = U2 + dU2dt_predictor * dt
    U3_pred = U3 + dU3dt_predictor * dt
    U5_pred = U5 + dU5dt_predictor * dt

    u_pred, v_pred, T_pred, rho_pred, p_pred, e_pred = primitive_variables(U1_pred, U2_pred, U3_pred, U5_pred)
    E_total_pred = U5_pred.copy()

    # Apply the boundary conditions
    u_pred, v_pred, T_pred, rho_pred, p_pred, e_pred = boundary_conditions(u_pred, v_pred, T_pred, rho_pred, p_pred,
                                                                           e_pred, boundary_type)

    """
    CORRECTOR STEP
    """

    tau_xx_E = tau_xx_E_forward(u_pred, v_pred, T_pred)
    tau_xy_E = tau_xy_E_forward(u_pred, v_pred, T_pred)
    tau_xy_F = tau_xy_F_forward(u_pred, v_pred, T_pred)
    tau_yy_F = tau_yy_F_forward(u_pred, v_pred, T_pred)
    qx_E = qx_E_forward(T_pred)
    qy_F = qy_F_forward(T_pred)
    E1, E2, E3, E5 = compute_E(u_pred, v_pred, rho_pred, p_pred, E_total_pred, tau_xx_E, tau_xy_E, qx_E)
    F1, F2, F3, F5 = compute_F(u_pred, v_pred, rho_pred, p_pred, E_total_pred, tau_yy_F, tau_xy_F, qy_F)

    dU1dt_corrector = np.zeros(shape=(N_Y, N_X))
    dU2dt_corrector = np.zeros(shape=(N_Y, N_X))
    dU3dt_corrector = np.zeros(shape=(N_Y, N_X))
    dU5dt_corrector = np.zeros(shape=(N_Y, N_X))
    for i in range(1, N_Y - 1):
        for j in range(1, N_X - 1):
            dU1dt_corrector[i, j] = -(E1[i, j] - E1[i, j - 1]) / DELTA_X - (F1[i, j] - F1[i + 1, j]) / DELTA_Y
            dU2dt_corrector[i, j] = -(E2[i, j] - E2[i, j - 1]) / DELTA_X - (F2[i, j] - F2[i + 1, j]) / DELTA_Y
            dU3dt_corrector[i, j] = -(E3[i, j] - E3[i, j - 1]) / DELTA_X - (F3[i, j] - F3[i + 1, j]) / DELTA_Y
            dU5dt_corrector[i, j] = -(E5[i, j] - E5[i, j - 1]) / DELTA_X - (F5[i, j] - F5[i + 1, j]) / DELTA_Y

    """
    FINAL STEP
    """

    # Determine the average derivative for the derived variables U1, U2, U3, U5
    dU1dt = 0.5 * (dU1dt_predictor + dU1dt_corrector)
    dU2dt = 0.5 * (dU2dt_predictor + dU2dt_corrector)
    dU3dt = 0.5 * (dU3dt_predictor + dU3dt_corrector)
    dU5dt = 0.5 * (dU5dt_predictor + dU5dt_corrector)

    # Update the values of the derived variables based on this average derivative
    U1 += dU1dt * dt
    U2 += dU2dt * dt
    U3 += dU3dt * dt
    U5 += dU5dt * dt

    # Calculate the primitive flow variables
    u, v, T, rho, p, e = primitive_variables(U1, U2, U3, U5)

    # Apply the boundary conditions
    u, v, T, rho, p, e = boundary_conditions(u, v, T, rho, p, e, boundary_type)

    return u, v, T, rho, p, e, dt
