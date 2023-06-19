import numpy as np
from nozzle_flow.conservation_form.constants import dx, N
from nozzle_flow.conservation_form.geometry import nozzle_area
from nozzle_flow.conservation_form.variable_conversion import get_rho, get_V, get_T
from nozzle_flow.conservation_form.initial_and_boundary_conditions import initial_conditions


def test_initial_conditions_primitive_variables():
    U1, U2, U3 = initial_conditions()

    # Create an array for the nozzle area
    A = np.array([nozzle_area(i * dx) for i in range(N)])

    # Test density
    rho = get_rho(A, U1)
    assert abs(rho[0] - 1.000) < 1e-3
    assert abs(rho[1] - 1.000) < 1e-3
    assert abs(rho[-2] - 0.091) < 1e-3
    assert abs(rho[-1] - 0.052) < 1e-3

    # Test speed
    V = get_V(U1, U2)
    assert abs(V[0] - 0.099) < 1e-3
    assert abs(V[1] - 0.111) < 1e-3
    assert abs(V[-2] - 1.221) < 1e-3
    assert abs(V[-1] - 1.901) < 1e-3

    # Test temperature
    T = get_T(U1, U2, U3)
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[1] - 1.000) < 1e-3
    assert abs(T[-2] - 0.342) < 1e-3
    assert abs(T[-1] - 0.307) < 1e-3
