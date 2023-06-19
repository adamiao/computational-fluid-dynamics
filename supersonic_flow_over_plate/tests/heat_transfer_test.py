from supersonic_flow_over_plate.initial_conditions import initial_conditions
from supersonic_flow_over_plate.physics import compute_thermal_conductivity
from supersonic_flow_over_plate.heat_transfer import (
    qx_E_backward,
    qx_E_forward,
    qy_F_backward,
    qy_F_forward,
)
from supersonic_flow_over_plate.constants import (
    TEMPERATURE_E,
    TEMPERATURE_WALL,
    DELTA_Y,
)


def test_qx_E_backward():
    u, v, T, p, rho, e = initial_conditions()
    qx = qx_E_backward(T)
    assert abs(sum(qx[-1, :]) / len(qx[-1, :]) - 0.0) < 1e-9
    assert abs(sum(qx[-2, :]) / len(qx[-2, :]) - 0.0) < 1e-9


def test_qx_E_forward():
    u, v, T, p, rho, e = initial_conditions()
    qx = qx_E_forward(T)
    assert abs(sum(qx[-1, :]) / len(qx[-1, :]) - 0.0) < 1e-9
    assert abs(sum(qx[-2, :]) / len(qx[-2, :]) - 0.0) < 1e-9


def test_qy_F_backward():
    DELTA_TEMPERATURE = 100
    u, v, T, p, rho, e = initial_conditions(temperature=TEMPERATURE_E,
                                            temperature_wall=TEMPERATURE_WALL - DELTA_TEMPERATURE)
    qy = qy_F_backward(T)
    k1 = compute_thermal_conductivity(TEMPERATURE_WALL - DELTA_TEMPERATURE)
    k2 = compute_thermal_conductivity(TEMPERATURE_E)
    assert abs(qy[-1, 0] - 0.0) < 1e-9
    assert abs(qy[-2, 0] - 0.0) < 1e-9
    assert abs(sum(qy[-1, 1:]) / len(qy[-1, 1:]) + k1 * DELTA_TEMPERATURE / DELTA_Y) < 1e-7
    assert abs(sum(qy[-2, 1:]) / len(qy[-2, 1:]) + k2 * DELTA_TEMPERATURE / DELTA_Y) < 1e-7


def test_qy_F_forward():
    DELTA_TEMPERATURE = 100
    u, v, T, p, rho, e = initial_conditions(temperature=TEMPERATURE_E,
                                            temperature_wall=TEMPERATURE_WALL - DELTA_TEMPERATURE)
    qy = qy_F_forward(T)
    k = compute_thermal_conductivity(TEMPERATURE_WALL - DELTA_TEMPERATURE)
    assert abs(qy[-1, 0] - 0.0) < 1e-9
    assert abs(qy[-2, 0] - 0.0) < 1e-9
    assert abs(sum(qy[-1, 1:]) / len(qy[-1, 1:]) + k * DELTA_TEMPERATURE / DELTA_Y) < 1e-7
    assert abs(sum(qy[-2, 1:]) / len(qy[-2, 1:]) - 0.0) < 1e-7
