from nozzle_flow.shock_capturing.constants import GAMMA

"""
PRIMITIVE VARIABLES
"""


def get_rho(A, U1):
    """
    Calculates the numpy array for the primitive variable 'density'
    Args:
        A: numpy array for the nozzle area
        U1: numpy array for the derived variable 'U1'

    Returns: numpy array
    """
    return U1 / A


def get_V(U1, U2):
    """
    Calculates the numpy array for the primitive variable 'speed'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'

    Returns: numpy array
    """
    return U2 / U1


def get_T(U1, U2, U3):
    """
    Calculates the numpy array for the primitive variable 'temperature'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: numpy array
    """
    return (GAMMA - 1) * (U3 / U1 - GAMMA / 2 * (U2 / U1) ** 2)


def get_P(A, U1, U2, U3):
    """
    Calculates the numpy array for the primitive variable 'pressure'
    Args:
        A: numpy array for the nozzle area
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: numpy array
    """
    return get_rho(A, U1) * get_T(U1, U2, U3)


def get_M(U1, U2, U3):
    """
    Calculates the numpy array for the primitive variable 'Mach number'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: numpy array
    """
    return get_V(U1, U2) / (get_T(U1, U2, U3) ** 0.5)


def get_mass_flow(U2):
    """
    Calculates the numpy array for the primitive variable 'mass flow'
    Args:
        U2: numpy array for the derived variable 'U2'

    Returns: numpy array
    """
    return U2


"""
DERIVED VARIABLES
"""


def get_U1(rho, A):
    """
    Calculates the numpy array for the derived variable 'U1'
    Args:
        rho: numpy array for the density
        A: numpy array for the nozzle area

    Returns: numpy array
    """
    return rho * A


def get_U2(rho, A, V):
    """
    Calculates the numpy array for the derived variable 'U2'
    Args:
        rho: numpy array for the density
        A: numpy array for the nozzle area
        V: numpy array for the speed

    Returns: numpy array
    """
    return rho * A * V


def get_U3(rho, A, V, T):
    """
    Calculates the numpy array for the derived variable 'U3'
    Args:
        rho: numpy array for the density
        A: numpy array for the nozzle area
        V: numpy array for the speed
        T: numpy array for the temperature

    Returns: numpy array
    """
    return rho * (T / (GAMMA - 1) + GAMMA / 2 * V ** 2) * A


def get_F1(U2):
    """
    Calculates the numpy array for the derived variable 'F1'
    Args:
        U2: numpy array for the derived variable 'U2'

    Returns: numpy array
    """
    return U2


def get_F2(U1, U2, U3):
    """
    Calculates the numpy array for the derived variable 'F2'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: numpy array
    """
    return U2 ** 2 / U1 + (GAMMA - 1) / GAMMA * (U3 - GAMMA / 2 * U2 ** 2 / U1)


def get_F3(U1, U2, U3):
    """
    Calculates the numpy array for the derived variable 'F3'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'

    Returns: numpy array
    """
    return GAMMA * U2 * U3 / U1 - GAMMA * (GAMMA - 1) / 2 * U2 ** 3 / (U1 ** 2)


def get_J2(U1, U2, U3, d_ln_A_dx):
    """
    Calculates the numpy array for the derived variable 'J2'
    Args:
        U1: numpy array for the derived variable 'U1'
        U2: numpy array for the derived variable 'U2'
        U3: numpy array for the derived variable 'U3'
        d_ln_A_dx: numpy array

    Returns: numpy array
    """
    return (GAMMA - 1) / GAMMA * (U3 - GAMMA / 2 * U2 ** 2 / U1) * d_ln_A_dx
