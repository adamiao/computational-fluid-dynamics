from expansion_wave.physics import (
    prandtl_meyer_function,
    inverse_prandtl_meyer_function,
    mach_angle,
)


def test_prandtl_meyer_function():
    mach_numbers = [1.0, 2.34, 11.33, 34.98, 98.12]
    for mach_number in mach_numbers:
        assert abs(mach_number - inverse_prandtl_meyer_function(prandtl_meyer_function(mach_number))) < 1e-3


def test_mach_angle():
    assert abs(mach_angle(0, False) - 1.571) < 1e-3
    assert abs(mach_angle(0, True) - 90.0) < 1e-3
    assert abs(mach_angle(1, False) - 0.785) < 1e-3
    assert abs(mach_angle(1, True) - 45.0) < 1e-3
