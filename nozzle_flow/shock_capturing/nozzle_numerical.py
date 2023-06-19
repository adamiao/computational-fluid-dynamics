import numpy as np
from nozzle_flow.shock_capturing.constants import ITERATIONS, N, dx
from nozzle_flow.shock_capturing.geometry import nozzle_area
from nozzle_flow.shock_capturing.variable_conversion import get_rho, get_V, get_T, get_P, get_M, get_mass_flow
from nozzle_flow.shock_capturing.initial_and_boundary_conditions import (
    initial_conditions,
    enforce_boundary_conditions,
)
from nozzle_flow.shock_capturing.predictor_corrector_step import (
    calculate_dt,
    predictor_step,
    corrector_step,
    calculate_next_iteration_variables,
)


def run_solver(U1, U2, U3):

    # Calculate the step time in the time-marching solution
    dt = calculate_dt(U1, U2, U3)

    # Calculate the 'predictor' step of the algorithm
    pred_U1, pred_U2, pred_U3, drho_dt, du_dt, dT_dt = predictor_step(U1, U2, U3, dt)

    # Calculate the 'corrector' step of the algorithm
    pred_dU1_dt, pred_dU2_dt, pred_dU3_dt, pred_S1, pred_S2, pred_S3 = corrector_step(pred_U1, pred_U2, pred_U3)

    # Calculate the next iteration of variables for the flow
    calculate_next_iteration_variables(U1, U2, U3, drho_dt, du_dt, dT_dt, pred_dU1_dt, pred_dU2_dt, pred_dU3_dt,
                                       pred_S1, pred_S2, pred_S3, dt)

    # Enforce boundary conditions
    enforce_boundary_conditions(U1, U2, U3)


def run_solver_n_iterations(U1, U2, U3, n_iterations):
    for _ in range(n_iterations):
        run_solver(U1, U2, U3)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Initialize fluid variables
    U1, U2, U3 = initial_conditions()

    # Run main solver for multiple iterations
    run_solver_n_iterations(U1, U2, U3, n_iterations=ITERATIONS)

    # Calculate the nozzle area and the nozzle longitudinal position
    A = np.array([nozzle_area(i * dx) for i in range(N)])
    x = np.array([i * dx for i in range(N)])

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

    # Plot the nondimensional pressure as a function of longitudinal position
    plt.scatter(x, pressure_field, c=rho, cmap='summer', edgecolors='black', linewidth=1, alpha=0.75)
    cbar = plt.colorbar()
    cbar.set_label('Nondimensional Density')

    plt.title('Nondimensional Pressure Field through the nozzle')
    plt.xlabel(r'$x / L$')
    plt.ylabel(r'$p / p_0$')

    plt.tight_layout()

    plt.show()
