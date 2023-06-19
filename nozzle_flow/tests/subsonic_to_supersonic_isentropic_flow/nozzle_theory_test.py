from nozzle_flow.subsonic_to_supersonic_isentropic_flow.nozzle_theory import (
    pressure_ratio,
    density_ratio,
    temperature_ratio,
)


def test_pressure_ratio():
    assert abs(pressure_ratio(1) - 0.528) < 1e-3


def test_density_ratio():
    assert abs(density_ratio(1) - 0.634) < 1e-4


def test_temperature_ratio():
    assert abs(temperature_ratio(1) - 0.833) < 1e-3
