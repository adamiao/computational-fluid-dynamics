import numpy as np
from nozzle_flow.shock_capturing.constants import GAMMA, dx, xN, N, U2_INITIAL_VALUE, PRESSURE_RATIO_AT_EXIT
from nozzle_flow.shock_capturing.geometry import nozzle_area
from nozzle_flow.shock_capturing.variable_conversion import get_V


def initial_conditions():
    U1, U2, U3 = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        U2[i] = U2_INITIAL_VALUE
        A = nozzle_area(i * dx)
        if 0 <= i * dx <= 0.5:
            rho = 1
            T = 1
        elif 0.5 < i * dx <= 1.5:
            rho = 1.0 - 0.366 * (i * dx - 0.5)
            T = 1.0 - 0.167 * (i * dx - 0.5)
        elif 1.5 < i * dx <= 2.1:
            rho = 0.634 - 0.702 * (i * dx - 1.5)
            T = 0.833 - 0.4908 * (i * dx - 1.5)
        else:
            rho = 0.5892 + 0.10228 * (i * dx - 2.1)
            T = 0.93968 + 0.0622 * (i * dx - 2.1)

        V = U2_INITIAL_VALUE / (rho * A)
        U1[i] = rho * A
        U3[i] = rho * (T / (GAMMA - 1) + GAMMA / 2 * V ** 2) * A

    return U1, U2, U3


def enforce_boundary_conditions(U1, U2, U3):
    """
    OBS: Not that we must change the exit boundary conditions since in this application we know that the flow will be
    subsonic.
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: makes an inplace modification of the input
    """

    # Determine the speed from the derived variables 'U1' and 'U2'
    V = get_V(U1, U2)

    # Nozzle Entry (the density and the temperature are both equal to 1 at the entry)
    U1[0] = nozzle_area(0)
    U2[0] = 2 * U2[1] - U2[2]
    U3[0] = U1[0] * (1 / (GAMMA - 1) + GAMMA / 2 * V[0] ** 2)  # U3 at index 0 using the value of the speed at index 0

    # Nozzle Exit
    U1[-1] = 2 * U1[-2] - U1[-3]
    U2[-1] = 2 * U2[-2] - U2[-3]
    U3[-1] = PRESSURE_RATIO_AT_EXIT * nozzle_area(xN) / (GAMMA - 1) + GAMMA / 2 * U2[-1] * V[-1]
