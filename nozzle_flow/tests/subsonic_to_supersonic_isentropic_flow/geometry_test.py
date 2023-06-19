import numpy as np
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.constants import x0, dx, N
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.geometry import nozzle_area


def test_nozzle_area():
    area = np.array([nozzle_area(x0 + i * dx) for i in range(N)])

    assert abs(area[0] - 5.9500) < 1e-3
    assert abs(area[1] - 5.3120) < 1e-3
    assert abs(area[-2] - 5.312) < 1e-3
    assert abs(area[-1] - 5.950) < 1e-3
