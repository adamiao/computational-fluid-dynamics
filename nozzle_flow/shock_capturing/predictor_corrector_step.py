import numpy as np
from nozzle_flow.shock_capturing.geometry import nozzle_area
from nozzle_flow.shock_capturing.constants import N, dx, C, Cx
from nozzle_flow.shock_capturing.variable_conversion import (
    get_V,
    get_T,
    get_P,
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


def artificial_viscosity(U1, U2, U3):

    # Calculate the nozzle area
    A = np.array([nozzle_area(i * dx) for i in range(N)])

    # Calculate pressure distribution
    P = get_P(A, U1, U2, U3)

    # Calculate the three components of the artificial viscosity (associated with U1, U2, and U3 respectively)
    S1, S2, S3 = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(1, N - 1):
        pressure_term = np.abs(P[i + 1] - 2 * P[i] + P[i - 1]) / (P[i + 1] + 2 * P[i] + P[i - 1])
        S1[i] = Cx * pressure_term * (U1[i + 1] - 2 * U1[i] + U1[i - 1])
        S2[i] = Cx * pressure_term * (U2[i + 1] - 2 * U2[i] + U2[i - 1])
        S3[i] = Cx * pressure_term * (U3[i + 1] - 2 * U3[i] + U3[i - 1])

    return S1, S2, S3


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

    # Calculate the artificial viscosity terms
    S1, S2, S3 = artificial_viscosity(U1, U2, U3)

    # Calculate the predicted value for all the derived variables
    pred_U1, pred_U2, pred_U3 = U1.copy(), U2.copy(), U3.copy()  # copy the numpy arrays to not modify mutable objects
    pred_U1 += dU1_dt * dt + S1
    pred_U2 += dU2_dt * dt + S2
    pred_U3 += dU3_dt * dt + S3

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

    # Calculate the predicted artificial viscosity terms because they will be used in the last update of U1, U2, U3
    # (namely within the function 'calculate_next_iteration_variables')
    pred_S1, pred_S2, pred_S3 = artificial_viscosity(pred_U1, pred_U2, pred_U3)

    return pred_dU1_dt, pred_dU2_dt, pred_dU3_dt, pred_S1, pred_S2, pred_S3


def calculate_next_iteration_variables(U1, U2, U3, dU1_dt, dU2_dt, dU3_dt, pred_dU1_dt, pred_dU2_dt, pred_dU3_dt,
                                       pred_S1, pred_S2, pred_S3, dt):

    # Calculate the value of the derived variables for the next time step as well as the average derivative using the
    # results from the predictor-corrector steps
    dU1_dt_avg = 0.5 * (dU1_dt + pred_dU1_dt)
    U1 += dU1_dt_avg * dt + pred_S1

    dU2_dt_avg = 0.5 * (dU2_dt + pred_dU2_dt)
    U2 += dU2_dt_avg * dt + pred_S2

    dU3_dt_avg = 0.5 * (dU3_dt + pred_dU3_dt)
    U3 += dU3_dt_avg * dt + pred_S3
