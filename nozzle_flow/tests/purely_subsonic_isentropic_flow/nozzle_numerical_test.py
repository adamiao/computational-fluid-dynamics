"""
This test will run the entire solver for 5000 iteration and compare it to the results showed by the book:

    Computational Fluid Dynamics - The Basics with Applications
    John D. Anderson, Jr.

Check the results on page 332, in chapter 7.
"""

from nozzle_flow.purely_subsonic_isentropic_flow.constants import ITERATIONS
from nozzle_flow.purely_subsonic_isentropic_flow.nozzle_numerical import run_solver_n_iterations
from nozzle_flow.purely_subsonic_isentropic_flow.initial_and_boundary_conditions import initial_conditions
from nozzle_flow.purely_subsonic_isentropic_flow.predictor_corrector_step import (
    calculate_mach_number,
    calculate_pressure,
    calculate_mass_flow,
)


def test_run_solver_5000_iterations():

    # Initialize fluid variables
    rho, u, T, p, M = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(rho, u, T, n_iterations=ITERATIONS)

    # Calculate the pressure field
    pressure_field = calculate_pressure(rho, T)

    # Calculate Mach number for the grid
    mach_number = calculate_mach_number(u, T)

    # Calculate mass flow
    m_dot = calculate_mass_flow(rho, u)

    # Test density
    assert abs(rho[0] - 1.000) < 1e-3
    assert abs(rho[15] - 0.861) < 1e-3
    assert abs(rho[-1] - 0.950) < 1e-3

    # Test speed
    assert abs(u[0] - 0.078) < 1e-3
    assert abs(u[15] - 0.543) < 1e-3
    assert abs(u[-1] - 0.328) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.000) < 1e-3
    assert abs(T[15] - 0.942) < 1e-3
    assert abs(T[-1] - 0.979) < 1e-3

    # Test pressure and Mach number
    assert abs(pressure_field[15] - 0.811) < 1e-3
    assert abs(pressure_field[-1] - 0.93) < 1e-3
    assert abs(mach_number[15] - 0.560) < 1e-3
    assert abs(mach_number[-1] - 0.331) < 1e-3
    assert abs(m_dot[15] - 0.468) < 1e-3
    assert abs(m_dot[-1] - 0.467) < 1e-3
