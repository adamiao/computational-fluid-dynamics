import numpy as np
from nozzle_flow.conservation_form.constants import ITERATIONS, N, dx
from nozzle_flow.conservation_form.geometry import nozzle_area
from nozzle_flow.conservation_form.variable_conversion import get_rho, get_V, get_T, get_P, get_M, get_mass_flow
from nozzle_flow.conservation_form.initial_and_boundary_conditions import (
    initial_conditions,
    enforce_boundary_conditions,
)
from nozzle_flow.conservation_form.predictor_corrector_step import (
    calculate_dt,
    predictor_step,
    corrector_step,
    calculate_next_iteration_variables,
)


def run_solver(U1, U2, U3):

    # Calculate the step time in the time-marching solution
    dt = calculate_dt(U1, U2, U3)

    # Calculate the 'predictor' step of the algorithm
    pred_U1, pred_U2, pred_U3, dU1_dt, dU2_dt, dU3_dt = predictor_step(U1, U2, U3, dt)

    # Calculate the 'corrector' step of the algorithm
    pred_dU1_dt, pred_dU2_dt, pred_dU3_dt = corrector_step(pred_U1, pred_U2, pred_U3)

    # Calculate the next iteration of variables for the flow
    calculate_next_iteration_variables(U1, U2, U3, dU1_dt, dU2_dt, dU3_dt, pred_dU1_dt, pred_dU2_dt, pred_dU3_dt, dt)

    # Enforce boundary conditions
    enforce_boundary_conditions(U1, U2, U3)


def run_solver_n_iterations(U1, U2, U3, n_iterations):
    for _ in range(n_iterations):
        run_solver(U1, U2, U3)


if __name__ == '__main__':

    # Initialize fluid variables
    U1, U2, U3 = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(U1, U2, U3, n_iterations=ITERATIONS)

    # Calculate the nozzle area
    A = np.array([nozzle_area(i * dx) for i in range(N)])

    # Calculate the density
    rho = get_rho(A, U1)

    # Calculate the speed
    V = get_V(U1, U2)

    # Calculate the temperature
    T = get_T(U1, U2, U3)

    # Calculate the pressure field
    pressure_field = get_P(A, U1, U2, U3)

    # Calculate Mach number for the grid
    mach_number = get_M(U1, U2, U3)

    # Calculate mass flow
    m_dot = get_mass_flow(U2)
