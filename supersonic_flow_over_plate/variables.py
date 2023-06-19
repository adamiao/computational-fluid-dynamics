from supersonic_flow_over_plate.physics import compute_viscosity
from supersonic_flow_over_plate.constants import (
    N_X, N_Y, DELTA_X, DELTA_Y,
    PRANDTL, COURANT,
    GAMMA, CV, R,
)


def primitive_variables(U1, U2, U3, U5):
    """
    Get the primitive flow variables from the derived variable U. Obs: U4 is related to the third spatial dimension and
    will not be used for this problem.
    Args:
        U1: numpy array
        U2: numpy array
        U3: numpy array
        U5: numpy array

    Returns: tuple of numpy arrays representing the primitive variables in the following order:
    x-speed, y-speed, temperature, density, pressure, specific energy
    """

    rho = U1
    u = U2 / U1
    v = U3 / U1
    e = U5 / U1 - 0.5 * ((U2 / U1) ** 2 + (U3 / U1) ** 2)
    T = e / CV
    p = rho * R * T

    return u, v, T, rho, p, e


def compute_total_energy(u, v, T, rho):
    """
    Computes the total energy field of the flow.
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        T: numpy array representing the temperature field
        rho: numpy array representing the density field

    Returns: numpy array representing the total energy of the flow
    """
    return rho * (CV * T + 0.5 * (u ** 2 + v ** 2))


def compute_U(u, v, T, rho):
    """
    Computation of the derived variables U1, U2, U3, U5
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        T: numpy array representing the temperature field
        rho: numpy array representing the density field

    Returns: tuple of numpy arrays representing the derived variables U1, U2, U3, U5
    """

    U1 = rho
    U2 = rho * u
    U3 = rho * v
    U5 = rho * (CV * T + 0.5 * (u ** 2 + v ** 2))
    return U1, U2, U3, U5


def compute_E(u, v, rho, p, E_total, tau_xx, tau_xy, qx):
    """
    Computation of the derived variables E1, E2, E3, E5
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        rho: numpy array representing the density field
        p: numpy array representing the pressure field
        E_total: numpy array representing the total energy field
        tau_xx: numpy array representing the shear stress field tau_xx
        tau_xy: numpy array representing the shear stress field tau_xy
        qx: numpy array representing the heat transfer in the x-direction

    Returns: tuple of numpy arrays representing the derived variables E1, E2, E3, E5
    """

    E1 = rho * u
    E2 = rho * u ** 2 + p - tau_xx
    E3 = rho * u * v - tau_xy
    E5 = (E_total + p) * u - u * tau_xx - v * tau_xy + qx
    return E1, E2, E3, E5


def compute_F(u, v, rho, p, E_total, tau_yy, tau_xy, qy):
    """
    Computation of the derived variables F1, F2, F3, F5
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        rho: numpy array representing the density field
        p: numpy array representing the pressure field
        E_total: numpy array representing the total energy field
        tau_yy: numpy array representing the shear stress field tau_yy
        tau_xy: numpy array representing the shear stress field tau_xy
        qy: numpy array representing the heat transfer in the y-direction

    Returns: tuple of numpy arrays representing the derived variables F1, F2, F3, F5
    """

    F1 = rho * v
    F2 = rho * u * v - tau_xy
    F3 = rho * v ** 2 + p - tau_yy
    F5 = (E_total + p) * v - u * tau_xy - v * tau_yy + qy
    return F1, F2, F3, F5


def time_step(u, v, T, rho):
    """
    Calculation of the time step for updating the flow variables
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        T: numpy array representing the temperature field
        rho: numpy array representing the density field

    Returns: float
    """

    # Determine maximum parameter 'nu' that will be used in the final equation for 'dt'
    nu = -float('inf')
    nu_factor = max(4 / 3, GAMMA / PRANDTL)
    for i in range(1, N_Y - 1):
        for j in range(1, N_X - 1):
            mu = compute_viscosity(T[i, j])
            temp_nu = nu_factor * mu / rho[i, j]
            nu = max(nu, temp_nu)

    # Determine 'dt_cfl' that will be used in the final equation for 'dt'
    dt = float('inf')
    for i in range(1, N_Y - 1):
        for j in range(1, N_X - 1):
            a = (GAMMA * R * T[i, j]) ** 0.5
            dxdy = 1 / DELTA_X ** 2 + 1 / DELTA_Y ** 2
            term_0 = abs(u[i, j]) / DELTA_X
            term_1 = abs(v[i, j]) / DELTA_Y
            term_2 = a * dxdy ** 0.5
            term_3 = 2 * nu * dxdy
            dt_cfl = COURANT * (term_0 + term_1 + term_2 + term_3) ** -1
            dt = min(dt, dt_cfl)

    return dt
