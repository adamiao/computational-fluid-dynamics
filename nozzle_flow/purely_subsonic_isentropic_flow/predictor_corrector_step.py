import numpy as np
from nozzle_flow.purely_subsonic_isentropic_flow.geometry import nozzle_area
from nozzle_flow.purely_subsonic_isentropic_flow.constants import GAMMA, R, N, C, dx


def calculate_area():
    return np.array([nozzle_area(i * dx) for i in range(N)])


def calculate_dt(u, T):
    dt_grid = np.zeros(N)
    for i in range(N):
        dt_grid[i] = C * dx / (u[i] + T[i] ** 0.5)
    dt = min(dt_grid)
    return dt


def calculate_speed_of_sound(T):
    a = np.zeros(N)
    for i in range(N):
        a[i] = (GAMMA * R * T[i]) ** 0.5
    return a


def calculate_mach_number(u, T):
    return u / (T ** 0.5)


def calculate_pressure(rho, T):
    return rho * T


def calculate_mass_flow(rho, u):
    area = calculate_area()
    return rho * u * area


def predictor_step(rho, u, T, dt):

    # Calculate the derivatives for the predictor step of the algorithm
    drho_dt, du_dt, dT_dt = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(1, N - 1):
        area_i_plus_1, area_i = nozzle_area((i + 1) * dx), nozzle_area(i * dx)
        log_area_ratio = (np.log(area_i_plus_1) - np.log(area_i)) / dx

        # Density derivative
        term_0 = -rho[i] * (u[i + 1] - u[i]) / dx
        term_1 = -rho[i] * u[i] * log_area_ratio
        term_2 = -u[i] * (rho[i + 1] - rho[i]) / dx
        drho_dt[i] = term_0 + term_1 + term_2

        # Speed derivative
        term_0 = -u[i] * (u[i + 1] - u[i]) / dx
        term_1 = -1 / GAMMA * ((T[i + 1] - T[i]) / dx + T[i] / rho[i] * (rho[i + 1] - rho[i]) / dx)
        du_dt[i] = term_0 + term_1

        # Temperature derivative
        term_0 = -u[i] * (T[i + 1] - T[i]) / dx
        term_1_a = (u[i + 1] - u[i]) / dx
        term_1_b = u[i] * log_area_ratio
        term_1 = -(GAMMA - 1) * T[i] * (term_1_a + term_1_b)
        dT_dt[i] = term_0 + term_1

    # Calculate the predicted value for the fluid variables
    pred_rho, pred_u, pred_T = rho.copy(), u.copy(), T.copy()  # copy the numpy arrays to not modify mutable objects
    pred_rho += drho_dt * dt
    pred_u += du_dt * dt
    pred_T += dT_dt * dt

    return pred_rho, pred_u, pred_T, drho_dt, du_dt, dT_dt


def corrector_step(pred_rho, pred_u, pred_T):

    # Calculate the predicted derivatives for the fluid variables
    pred_drho_dt, pred_du_dt, pred_dT_dt = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(1, N - 1):
        area_i, area_i_minus_1 = nozzle_area(i * dx), nozzle_area((i - 1) * dx)
        log_area_ratio = (np.log(area_i) - np.log(area_i_minus_1)) / dx

        # Predicted density derivative
        term_0 = -pred_rho[i] * (pred_u[i] - pred_u[i - 1]) / dx
        term_1 = -pred_rho[i] * pred_u[i] * log_area_ratio
        term_2 = -pred_u[i] * (pred_rho[i] - pred_rho[i - 1]) / dx
        pred_drho_dt[i] = term_0 + term_1 + term_2

        # Predicted speed derivative
        term_0 = -pred_u[i] * (pred_u[i] - pred_u[i - 1]) / dx
        term_1_a = (pred_T[i] - pred_T[i - 1]) / dx
        term_1_b = pred_T[i] / pred_rho[i] * (pred_rho[i] - pred_rho[i - 1]) / dx
        term_1 = -1 / GAMMA * (term_1_a + term_1_b)
        pred_du_dt[i] = term_0 + term_1

        # Predicted temperature derivative
        term_0 = -pred_u[i] * (pred_T[i] - pred_T[i - 1]) / dx
        term_1_a = (pred_u[i] - pred_u[i - 1]) / dx
        term_1_b = pred_u[i] * log_area_ratio
        term_1 = -(GAMMA - 1) * pred_T[i] * (term_1_a + term_1_b)
        pred_dT_dt[i] = term_0 + term_1

    return pred_drho_dt, pred_du_dt, pred_dT_dt


def calculate_next_iteration_variables(rho, u, T, drho_dt, du_dt, dT_dt, pred_drho_dt, pred_du_dt, pred_dT_dt, dt):

    # Calculate the value of the fluid variables for the next time step as well as the average derivative using the
    # results from the predictor-corrector step
    drho_dt_avg = 0.5 * (drho_dt + pred_drho_dt)
    rho += drho_dt_avg * dt

    du_dt_avg = 0.5 * (du_dt + pred_du_dt)
    u += du_dt_avg * dt

    dT_dt_avg = 0.5 * (dT_dt + pred_dT_dt)
    T += dT_dt_avg * dt
