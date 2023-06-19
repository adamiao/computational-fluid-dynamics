import numpy as np
from expansion_wave.constants import H, L, N_X, X_EXPANSION, THETA
from expansion_wave.geometry import (
    height,
    xi_coordinate,
    eta_coordinate,
    derivative_eta_x,
    coordinate_systems,
)


def test_height():
    DELTA_X = L / (N_X - 1)
    x_axis = np.array([i * DELTA_X for i in range(N_X)])
    h = np.array([height(x) for x in x_axis])
    tangent_theta = np.tan(np.radians(THETA))

    assert abs(h[0] - H) < 1e-3
    assert abs(h[-1] - H - (L - X_EXPANSION) * tangent_theta) < 1e-3
    assert abs(height(0) - H) < 1e-3
    assert abs(height(X_EXPANSION) - H) < 1e-3
    assert abs(height(X_EXPANSION + 1) - H - tangent_theta) < 1e-3
    assert abs(height(L) - H - (L - X_EXPANSION) * tangent_theta) < 1e-3


def test_eta():
    assert abs(eta_coordinate(0, 0) - 0.0) < 1e-3
    assert abs(eta_coordinate(0, 20) - 0.5) < 1e-3
    assert abs(eta_coordinate(7.8, 20) - 0.5) < 1e-3
    assert abs(eta_coordinate(X_EXPANSION, 20) - 0.5) < 1e-3
    assert abs(eta_coordinate(14.5, 40) - 1.0) < 1e-3
    assert abs(eta_coordinate(34.3, 40 - height(34.3)) - 0.0) < 1e-3
    assert abs(eta_coordinate(65, 40 - height(65)) - 0.0) < 1e-3


def test_deta_dx():
    assert abs(derivative_eta_x(0, 0) - 0.0) < 1e-6
    assert abs(derivative_eta_x(X_EXPANSION, 20) - 0.0) < 1e-6
    xi, eta = xi_coordinate(12.11, 0.8076), eta_coordinate(12.11, 0.8076)
    assert abs(derivative_eta_x(xi, eta) - 0.002272) < 1e-6


def test_coordinate_systems():
    DELTA_X = L / (N_X - 1)
    X = np.array([i * DELTA_X for i in range(N_X)])
    _, y = coordinate_systems(X)
    assert abs(y[0, 0] - 0.0) < 1e-3
    assert abs(y[-1, 0] - H) < 1e-3
    assert abs(y[0, -1] + 5.154) < 1e-3
    assert abs(y[-1, -1] - H) < 1e-3
