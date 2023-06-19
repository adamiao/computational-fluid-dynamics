"""
This test will run the entire solver for two scenarios:

    - 1 iteration
    - 1400 iterations

and then compare it to the results showed by the book:

    Computational Fluid Dynamics - The Basics with Applications
    John D. Anderson, Jr.

Check the results on page 352/353, in chapter 7.
"""

import numpy as np
from nozzle_flow.conservation_form.constants import N, dx
from nozzle_flow.conservation_form.geometry import nozzle_area
from nozzle_flow.conservation_form.nozzle_numerical import run_solver_n_iterations
from nozzle_flow.conservation_form.initial_and_boundary_conditions import initial_conditions
from nozzle_flow.conservation_form.variable_conversion import (
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
    run_solver_n_iterations(U1, U2, U3, n_iterations=1)

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
    assert abs(rho[15] - 0.633) < 1e-3
    assert abs(rho[-1] - 0.061) < 1e-3

    # Test speed
    assert abs(V[0] - 0.099) < 1e-3
    assert abs(V[15] - 0.930) < 1e-3
    assert abs(V[-1] - 1.479) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.834) < 1e-3
    assert abs(T[-1] - 0.196) < 1e-3

    # Test pressure
    assert abs(pressure_field[0] - 1.000) < 1e-3
    assert abs(pressure_field[15] - 0.528) < 1e-3
    assert abs(pressure_field[-1] - 0.012) < 1e-3

    # Test Mach number
    assert abs(mach_number[0] - 0.099) < 1e-3
    assert abs(mach_number[15] - 1.019) < 1e-3
    assert abs(mach_number[-1] - 3.342) < 1e-3


def test_run_solver_1400_iterations():

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
    assert abs(rho[15] - 0.692) < 1e-3
    assert abs(rho[-1] - 0.053) < 1e-3

    # Test speed
    assert abs(V[0] - 0.098) < 1e-3
    assert abs(V[15] - 0.848) < 1e-3
    assert abs(V[-1] - 1.866) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.858) < 1e-3
    assert abs(T[-1] - 0.306) < 1e-3

    # Test pressure
    assert abs(pressure_field[0] - 1.000) < 1e-3
    assert abs(pressure_field[15] - 0.594) < 1e-3
    assert abs(pressure_field[-1] - 0.016) < 1e-3

    # Test Mach number
    assert abs(mach_number[0] - 0.098) < 1e-3
    assert abs(mach_number[15] - 0.915) < 1e-3
    assert abs(mach_number[-1] - 3.375) < 1e-3
