from supersonic_flow_over_plate.initial_conditions import initial_conditions
from supersonic_flow_over_plate.physics import compute_viscosity
from supersonic_flow_over_plate.shear_stress import (
    tau_xy_E_backward,
    tau_xy_E_forward,
    tau_xy_F_backward,
    tau_xy_F_forward,
    tau_xx_E_backward,
    tau_xx_E_forward,
    tau_yy_F_backward,
    tau_yy_F_forward,
)
from supersonic_flow_over_plate.constants import (
    U_E,
    TEMPERATURE_WALL,
    DELTA_Y,
)


def test_tau_xy_e_backward():
    u, v, T, p, rho, e = initial_conditions()
    tau_xy = tau_xy_E_backward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xy[-1, :]) / len(tau_xy[-1, :]) - mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xy[-2, :]) / len(tau_xy[-2, :]) - mu * U_E / (2 * DELTA_Y)) < 1e-9


def test_tau_xy_E_forward():
    u, v, T, p, rho, e = initial_conditions()
    tau_xy = tau_xy_E_forward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xy[-1, :]) / len(tau_xy[-1, :]) - mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xy[-2, :]) / len(tau_xy[-2, :]) - mu * U_E / (2 * DELTA_Y)) < 1e-9


def test_tau_xy_F_backward():
    u, v, T, p, rho, e = initial_conditions()
    tau_xy = tau_xy_F_backward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xy[-1, :]) / len(tau_xy[-1, :]) - mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xy[-2, :]) / len(tau_xy[-2, :]) - mu * U_E / DELTA_Y) < 1e-9


def test_tau_xy_F_forward():
    u, v, T, p, rho, e = initial_conditions()
    tau_xy = tau_xy_F_forward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xy[-1, :]) / len(tau_xy[-1, :]) - mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xy[-2, :]) / len(tau_xy[-2, :]) - 0.0) < 1e-9


def test_tau_xx_E_backward():
    u, v, T, p, rho, e = initial_conditions(y_speed=U_E)
    tau_xx = tau_xx_E_backward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xx[-1, :]) / len(tau_xx[-1, :]) + 2 / 3 * mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xx[-2, :]) / len(tau_xx[-2, :]) + 2 / 3 * mu * U_E / (2 * DELTA_Y)) < 1e-9


def test_tau_xx_E_forward():
    u, v, T, p, rho, e = initial_conditions(y_speed=U_E)
    tau_xx = tau_xx_E_forward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_xx[-1, :]) / len(tau_xx[-1, :]) + 2 / 3 * mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_xx[-2, :]) / len(tau_xx[-2, :]) + 2 / 3 * mu * U_E / (2 * DELTA_Y)) < 1e-9


def test_tau_yy_F_backward():
    u, v, T, p, rho, e = initial_conditions(y_speed=U_E)
    tau_yy = tau_yy_F_backward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_yy[-1, :]) / len(tau_yy[-1, :]) - 4 / 3 * mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_yy[-2, :]) / len(tau_yy[-2, :]) - 4 / 3 * mu * U_E / DELTA_Y) < 1e-9


def test_tau_yy_F_forward():
    u, v, T, p, rho, e = initial_conditions(y_speed=U_E)
    tau_yy = tau_yy_F_forward(u, v, T)
    mu = compute_viscosity(TEMPERATURE_WALL)
    assert abs(sum(tau_yy[-1, :]) / len(tau_yy[-1, :]) - 4 / 3 * mu * U_E / DELTA_Y) < 1e-9
    assert abs(sum(tau_yy[-2, :]) / len(tau_yy[-2, :]) - 0.0) < 1e-9
