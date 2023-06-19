import numpy as np
from nozzle_flow.purely_subsonic_isentropic_flow.constants import x0, dx, N
from nozzle_flow.purely_subsonic_isentropic_flow.geometry import nozzle_area


def test_nozzle_area():
    area = np.array([nozzle_area(x0 + i * dx) for i in range(N)])

    assert abs(area[0] - 5.9500) < 1e-3
    assert abs(area[1] - 5.3120) < 1e-3
    assert abs(area[-2] - 1.436) < 1e-3
    assert abs(area[-1] - 1.500) < 1e-3
