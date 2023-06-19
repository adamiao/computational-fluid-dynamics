import numpy as np
from couette_flow.implicit_solution.constants import U_E, N


def initial_conditions():
    u = np.zeros(N)
    u[-1] = U_E
    return u
