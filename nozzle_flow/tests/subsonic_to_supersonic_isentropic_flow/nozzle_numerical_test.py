"""
This test will run the entire solver for two scenarios:

    - 1 iteration
    - 1400 iterations

and then compare it to the results showed by the book:

    Computational Fluid Dynamics - The Basics with Applications
    John D. Anderson, Jr.

Check the results on page 314, in chapter 7.
"""

from nozzle_flow.subsonic_to_supersonic_isentropic_flow.nozzle_numerical import run_solver_n_iterations
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.initial_and_boundary_conditions import initial_conditions
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.predictor_corrector_step import (
    calculate_mach_number,
    calculate_pressure,
    calculate_mass_flow,
)


def test_run_solver_1_iteration():

    # Initialize fluid variables
    rho, u, T, p, M = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(rho, u, T, n_iterations=1)

    # Calculate the pressure field
    pressure_field = calculate_pressure(rho, T)

    # Calculate Mach number for the grid
    mach_number = calculate_mach_number(u, T)

    # Test density
    assert abs(rho[0] - 1.000) < 1e-3
    assert abs(rho[15] - 0.531) < 1e-3
    assert abs(rho[-1] - 0.066) < 1e-3

    # Test speed
    assert abs(u[0] - 0.111) < 1e-3
    assert abs(u[15] - 1.394) < 1e-3
    assert abs(u[-1] - 1.895) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.656) < 1e-3
    assert abs(T[-1] - 0.309) < 1e-3

    # Test pressure and Mach number
    assert abs(pressure_field[15] - 0.349) < 1e-3
    assert abs(mach_number[15] - 1.720) < 1e-3


def test_run_solver_1400_iterations():

    # Initialize fluid variables
    rho, u, T, p, M = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(rho, u, T, n_iterations=1400)

    # Calculate the pressure field
    pressure_field = calculate_pressure(rho, T)

    # Calculate Mach number for the grid
    mach_number = calculate_mach_number(u, T)

    # Calculate mass flow
    m_dot = calculate_mass_flow(rho, u)

    # Test density
    assert abs(rho[0] - 1.000) < 1e-3
    assert abs(rho[15] - 0.639) < 1e-3
    assert abs(rho[-1] - 0.053) < 1e-3

    # Test speed
    assert abs(u[0] - 0.099) < 1e-3
    assert abs(u[15] - 0.914) < 1e-3
    assert abs(u[-1] - 1.862) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.836) < 1e-3
    assert abs(T[-1] - 0.308) < 1e-3

    # Test pressure and Mach number
    assert abs(pressure_field[15] - 0.534) < 1e-3
    assert abs(mach_number[15] - 0.999) < 1e-3
    assert abs(m_dot[15] - 0.584) < 1e-3
