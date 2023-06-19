import numpy as np
from couette_flow.pressure_correction_solution.constants import (
    NX_P, NY_P,
    NX_U, NY_U,
    NX_V, NY_V,
    U_E, V_E,
)


def initial_conditions():
    """
    Generates the pressure, x-coordinate speed, and y-coordinate speed
    Returns: (numpy array, numpy array, numpy array)
    """
    # Initialize flow variables
    p = np.zeros(shape=(NY_P, NX_P))
    u = np.zeros(shape=(NY_U, NX_U))
    v = np.zeros(shape=(NY_V, NX_V))

    # Top plate has a constant speed
    u[0, :] = U_E

    # Initiate some velocity in the y-coordinate direction (middle of the computational grid)
    v[NY_V // 2, NX_V // 2] = V_E

    return p, u, v


def boundary_conditions(p, u, v):
    """
    The following boundary conditions were chosen to simulate as if the computational grid was infinite from left to
    right.
    Args:
        p: numpy array (pressure field)
        u: numpy array (x-coordinate speed)
        v: numpy array (y-coordinate speed)

    Returns: None (computations done in-place)
    """
    # Pressure boundary conditions
    p[:, 0] = p[:, 1].copy()
    p[:, -1] = p[:, -2].copy()

    # x-coordinate speed boundary conditions
    u[:, 0] = u[:, 1].copy()
    u[:, -1] = u[:, -2].copy()

    # y-coordinate speed boundary conditions
    v[:, 0] = v[:, 1].copy()
    v[:, -1] = v[:, -2].copy()
