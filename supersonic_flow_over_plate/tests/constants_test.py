from supersonic_flow_over_plate.constants import (
    REYNOLDS,
    GAMMA,
    DENSITY_E,
    DELTA_X,
    DELTA_Y,
    SPEED_OF_SOUND_E,
)


def test_constants():
    assert abs(REYNOLDS - 931.782) < 1e-3
    assert abs(GAMMA - 1.4) < 1e-3
    assert abs(DENSITY_E - 1.225) < 1e-3
    assert abs(DELTA_X - 1.449e-7) < 1e-10
    assert abs(DELTA_Y - 1.187e-7) < 1e-10
    assert abs(SPEED_OF_SOUND_E - 340.28) < 1e-1
