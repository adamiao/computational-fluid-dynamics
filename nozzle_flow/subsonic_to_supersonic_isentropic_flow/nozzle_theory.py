from nozzle_flow.subsonic_to_supersonic_isentropic_flow.constants import GAMMA


def area_ratio(m):
    exponent = (GAMMA + 1) / (GAMMA - 1)
    cte = 2 / (GAMMA + 1) * (1 + (GAMMA - 1) / 2 * m ** 2)
    ratio_squared = 1 / m ** 2 * cte ** exponent
    return ratio_squared


def mach_numbers_from_area_ratio(ratio):
    eps = 1e-3
    if ratio < 1:
        return -1, -1

    a, b, m_subsonic = 0, 1, 0.5
    while abs(a - b) > eps:
        temporary_ratio = area_ratio(m_subsonic) ** 0.5
        if temporary_ratio > ratio:
            a = m_subsonic
        elif temporary_ratio < ratio:
            b = m_subsonic
        m_subsonic = (a + b) / 2

    a, b, m_supersonic = 1, 100, 4
    while abs(a - b) > eps:
        temporary_ratio = area_ratio(m_supersonic) ** 0.5
        if temporary_ratio > ratio:
            b = m_supersonic
        elif temporary_ratio < ratio:
            a = m_supersonic
        m_supersonic = (a + b) / 2

    return m_subsonic, m_supersonic


def pressure_ratio(m):
    exponent = -GAMMA / (GAMMA - 1)
    p_p0 = 1 + (GAMMA - 1) / 2 * m ** 2
    return p_p0 ** exponent


def density_ratio(m):
    exponent = -1 / (GAMMA - 1)
    rho_rho0 = 1 + (GAMMA - 1) / 2 * m ** 2
    return rho_rho0 ** exponent


def temperature_ratio(m):
    t_t0 = 1 + (GAMMA - 1) / 2 * m ** 2
    return t_t0 ** -1
