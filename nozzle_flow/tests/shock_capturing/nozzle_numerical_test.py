"""
This test will run the entire solver for 1400 iterations and then compare it to the results showed by the book:

    Computational Fluid Dynamics - The Basics with Applications
    John D. Anderson, Jr.

Check the results on page 368/369, in chapter 7.
"""

import numpy as np
from nozzle_flow.shock_capturing.constants import N, dx
from nozzle_flow.shock_capturing.geometry import nozzle_area
from nozzle_flow.shock_capturing.nozzle_numerical import run_solver_n_iterations
from nozzle_flow.shock_capturing.initial_and_boundary_conditions import initial_conditions
from nozzle_flow.shock_capturing.variable_conversion import (
    get_rho,
    get_V,
    get_T,
    get_M,
    get_P,
)


def test_run_solver_1_iteration():

    # Initialize fluid variables
    U1, U2, U3 = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(U1, U2, U3, n_iterations=1400)

    # Calculate nozzle area
    A = np.array([nozzle_area(i * dx) for i in range(N)])

    # Calculate density
    rho = get_rho(A, U1)

    # Calculate speed
    V = get_V(U1, U2)

    # Calculate temperature
    T = get_T(U1, U2, U3)

    # Calculate the pressure field
    pressure_field = get_P(A, U1, U2, U3)

    # Calculate Mach number for the grid
    mach_number = get_M(U1, U2, U3)

    # Test density
    assert abs(rho[0] - 1.000) < 1e-3
    assert abs(rho[15] - 0.969) < 1e-3
    assert abs(rho[30] - 0.634) < 1e-3
    assert abs(rho[-1] - 0.700) < 1e-3

    # Test speed
    assert abs(V[0] - 0.098) < 1e-3
    assert abs(V[15] - 0.270) < 1e-3
    assert abs(V[30] - 0.921) < 1e-3
    assert abs(V[-1] - 0.152) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.987) < 1e-3
    assert abs(T[30] - 0.832) < 1e-3
    assert abs(T[-1] - 0.968) < 1e-3

    # Test pressure
    assert abs(pressure_field[0] - 1.000) < 1e-3
    assert abs(pressure_field[15] - 0.957) < 1e-3
    assert abs(pressure_field[30] - 0.528) < 1e-3
    assert abs(pressure_field[-1] - 0.678) < 1e-3

    # Test Mach number
    assert abs(mach_number[0] - 0.098) < 1e-3
    assert abs(mach_number[15] - 0.271) < 1e-3
    assert abs(mach_number[30] - 1.010) < 1e-3
    assert abs(mach_number[-1] - 0.155) < 1e-3
