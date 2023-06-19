import numpy as np
from nozzle_flow.conservation_form.geometry import nozzle_area
from nozzle_flow.conservation_form.constants import N, dx, C
from nozzle_flow.conservation_form.variable_conversion import (
    get_V,
    get_T,
    get_F1,
    get_F2,
    get_F3,
    get_J2,
)


def calculate_dt(U1, U2, U3):
    V = get_V(U1, U2)
    T = get_T(U1, U2, U3)
    dt_grid = C * dx / (V + T ** 0.5)
    dt = min(dt_grid)
    return dt


def predictor_step(U1, U2, U3, dt):

    # Calculate the derivatives for the predictor step of the algorithm
    dU1_dt, dU2_dt, dU3_dt = np.zeros(N), np.zeros(N), np.zeros(N)

    # Create the derivative of the natural logarithm of the nozzle area
    d_ln_A_dx = np.zeros(N)
    for i in range(1, N - 1):
        d_ln_A_dx[i] = (np.log(nozzle_area((i + 1) * dx)) - np.log(nozzle_area(i * dx))) / dx

    # Calculate the remaining derived variables
    F1 = get_F1(U2)
    F2 = get_F2(U1, U2, U3)
    J2 = get_J2(U1, U2, U3, d_ln_A_dx)
    F3 = get_F3(U1, U2, U3)
    for i in range(1, N - 1):
        # Variable 'U1' derivative
        dU1_dt[i] = -(F1[i + 1] - F1[i]) / dx

        # Variable 'U2' derivative
        dU2_dt[i] = -(F2[i + 1] - F2[i]) / dx + J2[i]

        # Variable 'U3' derivative
        dU3_dt[i] = -(F3[i + 1] - F3[i]) / dx

    # Calculate the predicted value for all the derived variables
    pred_U1, pred_U2, pred_U3 = U1.copy(), U2.copy(), U3.copy()  # copy the numpy arrays to not modify mutable objects
    pred_U1 += dU1_dt * dt
    pred_U2 += dU2_dt * dt
    pred_U3 += dU3_dt * dt

    return pred_U1, pred_U2, pred_U3, dU1_dt, dU2_dt, dU3_dt


def corrector_step(pred_U1, pred_U2, pred_U3):

    # Calculate the predicted values
    pred_F1 = get_F1(pred_U2)
    pred_F2 = get_F2(pred_U1, pred_U2, pred_U3)
    pred_F3 = get_F3(pred_U1, pred_U2, pred_U3)

    # Calculate the predicted derivatives for the derived variables
    pred_dU1_dt, pred_dU2_dt, pred_dU3_dt = np.zeros(N), np.zeros(N), np.zeros(N)

    # Create the derivative of the natural logarithm of the nozzle area
    d_ln_A_dx = np.zeros(N)
    for i in range(1, N - 1):
        d_ln_A_dx[i] = (np.log(nozzle_area(i * dx)) - np.log(nozzle_area((i - 1) * dx))) / dx

    # Calculate the last predicted derived variable
    pred_J2 = get_J2(pred_U1, pred_U2, pred_U3, d_ln_A_dx)

    for i in range(1, N - 1):
        # Predicted U1 derivative
        pred_dU1_dt[i] = -(pred_F1[i] - pred_F1[i - 1]) / dx

        # Predicted U2 derivative
        pred_dU2_dt[i] = -(pred_F2[i] - pred_F2[i - 1]) / dx + pred_J2[i]

        # Predicted U3 derivative
        pred_dU3_dt[i] = -(pred_F3[i] - pred_F3[i - 1]) / dx

    return pred_dU1_dt, pred_dU2_dt, pred_dU3_dt


def calculate_next_iteration_variables(U1, U2, U3, dU1_dt, dU2_dt, dU3_dt, pred_dU1_dt, pred_dU2_dt, pred_dU3_dt, dt):

    # Calculate the value of the derived variables for the next time step as well as the average derivative using the
    # results from the predictor-corrector steps
    dU1_dt_avg = 0.5 * (dU1_dt + pred_dU1_dt)
    U1 += dU1_dt_avg * dt

    dU2_dt_avg = 0.5 * (dU2_dt + pred_dU2_dt)
    U2 += dU2_dt_avg * dt

    dU3_dt_avg = 0.5 * (dU3_dt + pred_dU3_dt)
    U3 += dU3_dt_avg * dt
