import numpy as np
from nozzle_flow.purely_subsonic_isentropic_flow.constants import N, x0, dx


def _rho_initial(x):
    return 1.0 - 0.023 * x


def _T_initial(x):
    return 1.0 - 0.009333 * x


def _u_initial(x):
    return 0.05 + 0.11 * x


def _p_initial(x):
    return _rho_initial(x) * _T_initial(x)


def _mach_initial(x):
    return _u_initial(x) / (_T_initial(x) ** 0.5)


def initial_conditions():
    rho = np.array([_rho_initial(x0 + i * dx) for i in range(N)])
    u = np.array([_u_initial(x0 + i * dx) for i in range(N)])
    T = np.array([_T_initial(x0 + i * dx) for i in range(N)])
    p = np.array([_p_initial(x0 + i * dx) for i in range(N)])
    M = np.array([_mach_initial(x0 + i * dx) for i in range(N)])
    return rho, u, T, p, M


def enforce_boundary_conditions(rho, u, T, pressure_exit_ratio):
    """
    Differently than what was done in 'subsonic_to_supersonic_isentropic_flow', here we must ensure that the pressure at
    the exit of the nozzle stays constant (equal to some prescribed value). To accomplish that we must have that the
    product of the density and temperature be kept constant. The entry boundary conditions are kept just the same.
    Args:
        rho: density numpy array
        u: speed numpy array
        T: temperature numpy array
        pressure_exit_ratio: pressure ratio at the nozzle exit

    Returns:

    """

    # Nozzle Entry
    u[0] = 2 * u[1] - u[2]

    # Nozzle Exit
    rho[-1] = 2 * rho[-2] - rho[-3]
    u[-1] = 2 * u[-2] - u[-3]
    T[-1] = pressure_exit_ratio / rho[-1]
