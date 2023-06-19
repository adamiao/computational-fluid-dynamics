import numpy as np
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.predictor_corrector_step import calculate_speed_of_sound
from nozzle_flow.subsonic_to_supersonic_isentropic_flow.constants import N


def test_calculate_speed_of_sound():

    # Test speed of sound calculation for T = 293K
    speed_of_sound = calculate_speed_of_sound(np.array([293 for _ in range(N)]))
    assert abs(np.average(speed_of_sound) - 343.144) < 1e-3
