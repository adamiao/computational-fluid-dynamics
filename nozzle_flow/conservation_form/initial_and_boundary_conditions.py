import numpy as np
from nozzle_flow.conservation_form.constants import GAMMA, dx, N, U2_INITIAL_VALUE
from nozzle_flow.conservation_form.geometry import nozzle_area
from nozzle_flow.conservation_form.variable_conversion import get_V


def initial_conditions():
    U1, U2, U3 = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        U2[i] = U2_INITIAL_VALUE
        A = nozzle_area(i * dx)
        if 0 <= i * dx <= 0.5:
            rho = 1
            T = 1
        elif 0.5 < i * dx < 1.5:
            rho = 1.0 - 0.366 * (i * dx - 0.5)
            T = 1.0 - 0.167 * (i * dx - 0.5)
        else:
            rho = 0.634 - 0.3879 * (i * dx - 1.5)
            T = 0.833 - 0.3507 * (i * dx - 1.5)

        V = U2_INITIAL_VALUE / (rho * A)
        U1[i] = rho * A
        U3[i] = rho * (T / (GAMMA - 1) + GAMMA / 2 * V ** 2) * A

    return U1, U2, U3


def enforce_boundary_conditions(U1, U2, U3):
    """
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: makes an inplace modification of the input
    """

    # Nozzle Entry (the density and the temperature are both equal to 1 at the entry)
    U1[0] = nozzle_area(0)
    U2[0] = 2 * U2[1] - U2[2]
    V_at_0 = get_V(U1, U2)[0]  # Get the speed from the derived variables U1, U2, at index 0
    U3[0] = U1[0] * (1 / (GAMMA - 1) + GAMMA / 2 * V_at_0 ** 2)  # U3 at index 0 using the value of the speed at index 0

    # Nozzle Exit
    U1[-1] = 2 * U1[-2] - U1[-3]
    U2[-1] = 2 * U2[-2] - U2[-3]
    U3[-1] = 2 * U3[-2] - U3[-3]
