from expansion_wave.constants import GAMMA, R


def decode_primitive_variables(F1, F2, F3, F4):
    """
    Decodes the derived variables to get the primitive variables (density, speed in the x-direction, speed in the
    y-direction, pressure, and temperature)
    Args:
        F1: one dimensional numpy array
        F2: one dimensional numpy array
        F3: one dimensional numpy array
        F4: one dimensional numpy array

    Returns: a tuple of one dimensional numpy arrays associated with the density, speed in the x-direction, speed in
    the y-direction, pressure, and temperature respectively.
    """
    # Helper variables
    A = F3 ** 2 / (2 * F1) - F4
    B = GAMMA / (GAMMA - 1) * F1 * F2
    C = -(GAMMA + 1) / (2 * (GAMMA - 1)) * F1 ** 3

    # Decode primitive variables
    rho = (-B + (B ** 2 - 4 * A * C) ** 0.5) / (2 * A)
    u = F1 / rho
    v = F3 / F1
    p = F2 - u * F1
    T = p / (rho * R)

    return rho, u, v, p, T


def calculate_derived_variables(F1, F2, F3, F4):
    """
    Calculates the remaining derived variables G1, G2, G3, G4.
    Args:
        F1: one dimensional numpy array
        F2: one dimensional numpy array
        F3: one dimensional numpy array
        F4: one dimensional numpy array

    Returns: a tuple of one dimensional numpy arrays representing the remaining derived variables G1, G2, G3, G4
    respectively.
    """
    rho, _, _, _, _ = decode_primitive_variables(F1, F2, F3, F4)
    G1 = rho * F3 / F1
    G2 = F3
    G3 = rho * (F3 / F1) ** 2 + F2 - F1 ** 2 / rho
    G4 = rho * F3 * F4 / (F1 ** 2)

    return G1, G2, G3, G4
