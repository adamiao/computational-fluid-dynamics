from supersonic_flow_over_plate.constants import (
    U_E,
    V_E,
    TEMPERATURE_E,
    TEMPERATURE_WALL,
    PRESSURE_E,
    CV,
    R,
)


def boundary_conditions(u, v, T, rho, p, e, boundary_type='constant_temperature'):
    """
    Boundary conditions that are enforced to the flow.
    Args:
        u: numpy array representing the x-direction speed
        v: numpy array representing the y-direction speed
        T: numpy array representing the temperature field
        rho: numpy array representing the density field
        p: numpy array representing the density field
        e: numpy array representing the specific energy field
        boundary_type: can be either 'constant_temperature' or 'adiabatic'

    Returns: a tuple of numpy arrays representing, respectfully:
    x-speed, y-speed, temperature, density, pressure, specific energy
    """

    # Entry of the physical domain
    u[:-1, 0] = U_E
    v[:-1, 0] = V_E
    T[:, 0] = TEMPERATURE_E
    p[:, 0] = PRESSURE_E
    rho[:, 0] = p[:, 0] / (R * T[:, 0])
    e[:, 0] = CV * T[:, 0]

    # Upper boundary
    u[0, :] = U_E
    v[0, :] = V_E
    T[0, :] = TEMPERATURE_E
    p[0, :] = PRESSURE_E
    rho[0, :] = p[0, :] / (R * T[0, :])
    e[0, :] = CV * T[0, :]

    # Exit of the physical domain
    u[1:-1, -1] = 2 * u[1:-1, -2] - u[1:-1, -3]
    v[1:-1, -1] = 2 * v[1:-1, -2] - v[1:-1, -3]
    T[1:-1, -1] = 2 * T[1:-1, -2] - T[1:-1, -3]
    p[1:-1, -1] = 2 * p[1:-1, -2] - p[1:-1, -3]
    rho[1:-1, -1] = p[1:-1, -1] / (R * T[1:-1, -1])
    e[1:-1, -1] = CV * T[1:-1, -1]

    # Lower boundary (in this section we must know what type of boundary condition we will be enforcing)
    u[-1, :] = 0.0
    v[-1, :] = 0.0
    if boundary_type == 'adiabatic':
        T[-1, 1:] = T[-2, 1:]
    else:
        T[-1, 1:] = TEMPERATURE_WALL
    p[-1, 1:] = 2 * p[-2, 1:] - p[-3, 1:]
    rho[-1, 1:] = p[-1, 1:] / (R * T[-1, 1:])
    e[-1, 1:] = CV * T[-1, 1:]

    return u, v, T, rho, p, e
