from nozzle_flow.subsonic_to_supersonic_isentropic_flow.constants import ITERATIONS
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.initial_and_boundary_conditions import (
    initial_conditions,
    enforce_boundary_conditions,
)
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.predictor_corrector_step import (
    calculate_dt,
    predictor_step,
    corrector_step,
    calculate_next_iteration_variables,
    calculate_mach_number,
    calculate_pressure,
    calculate_mass_flow,
)


def run_solver(rho, u, T):

    # Calculate the step time in the time-marching solution
    dt = calculate_dt(u, T)

    # Calculate the 'predictor' step of the algorithm
    pred_rho, pred_u, pred_T, drho_dt, du_dt, dT_dt = predictor_step(rho, u, T, dt)

    # Calculate the 'corrector' step of the algorithm
    pred_drho_dt, pred_du_dt, pred_dT_dt = corrector_step(pred_rho, pred_u, pred_T)

    # Calculate the next iteration of variables for the flow
    calculate_next_iteration_variables(rho, u, T, drho_dt, du_dt, dT_dt, pred_drho_dt, pred_du_dt, pred_dT_dt, dt)

    # Enforce boundary conditions
    enforce_boundary_conditions(rho, u, T)


def run_solver_n_iterations(rho, u, T, n_iterations):
    for _ in range(n_iterations):
        run_solver(rho, u, T)


if __name__ == '__main__':

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
