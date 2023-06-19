def nozzle_area(x):
    """
    Differently than what is being implemented in 'subsonic_to_supersonic_isentropic_flow', the output of this function
    is actually the ratio of the area at 'x' and the area at the throat, A_t.
    Args:
        x: non-dimensional position at the nozzle (in other words, this is really 'x / L'

    Returns: non-dimensional nozzle area ratio relative to the throat (A / A_t)
    """
    if 0 <= x <= 1.5:
        return 1 + 2.2 * (x - 1.5) ** 2
    elif 1.5 < x <= 3:
        return 1 + 0.2223 * (x - 1.5) ** 2
