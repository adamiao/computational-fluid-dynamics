import numpy as np
from expansion_wave.constants import GAMMA, R


def density(p, T):
    return p / (R * T)


def mach_angle(M, in_degrees=False):
    if isinstance(M, np.ndarray):
        if in_degrees:
            mu = np.array([np.arctan(1 / value) * 180 / np.pi if value > 0.0 else 90.0 for value in M])
        else:
            mu = np.array([np.arctan(1 / value) if value > 0.0 else np.pi / 2 for value in M])
    else:
        mu = np.arctan(1 / M) if M > 0.0 else np.pi / 2
        if in_degrees:
            mu = mu * 180 / np.pi
    return mu


def prandtl_meyer_function(M):
    if M < 1:
        return 0
    c1 = ((GAMMA + 1) / (GAMMA - 1)) ** 0.5
    c2 = M ** 2 - 1
    c3 = ((GAMMA - 1) / (GAMMA + 1) * c2) ** 0.5
    return c1 * np.arctan(c3) - np.arctan(c2)


def pressure_ratio(M1, M2):
    exponent = GAMMA / (GAMMA - 1)
    numerator = 1 + (GAMMA - 1) / 2 * M1 ** 2
    denominator = 1 + (GAMMA - 1) / 2 * M2 ** 2
    return (numerator / denominator) ** exponent


def temperature_ratio(M1, M2):
    numerator = 1 + (GAMMA - 1) / 2 * M1 ** 2
    denominator = 1 + (GAMMA - 1) / 2 * M2 ** 2
    return numerator / denominator


def inverse_prandtl_meyer_function(f):
    a, b, M, eps = 1, 100, 50.5, 1e-3
    while abs(b - a) > eps:
        if prandtl_meyer_function(M) <= f:
            a = M
        else:
            b = M
        M = max((b - a) / 2, (b + a) / 2)
    return M
