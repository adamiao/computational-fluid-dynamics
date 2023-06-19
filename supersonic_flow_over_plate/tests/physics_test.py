from supersonic_flow_over_plate.physics import (
    compute_viscosity,
    compute_thermal_conductivity,
)


def test_compute_viscosity():
    assert abs(compute_viscosity(T=-30 + 273.15) - 1.579e-5) / 1.579e-5 < 1e-2
    assert abs(compute_viscosity(T=0 + 273.15) - 1.729e-5) / 1.729e-5 < 1e-2
    assert abs(compute_viscosity(T=25 + 273.15) - 1.849e-5) / 1.849e-5 < 1e-2
    assert abs(compute_viscosity(T=1000 + 273.15) - 4.826e-5) / 4.826e-5 < 1e-2


def test_compute_thermal_conductivity():
    assert abs(compute_thermal_conductivity(T=-30 + 273.15, Pr=0.7425, cp=1.004) - 2.134e-5) / 2.134e-5 < 1e-2
    assert abs(compute_thermal_conductivity(T=0 + 273.15, Pr=0.7362, cp=1.006) - 2.364e-5) / 2.364e-5 < 1e-2
    assert abs(compute_thermal_conductivity(T=25 + 273.15, Pr=0.7296, cp=1.007) - 2.551e-5) / 2.551e-5 < 1e-2
    assert abs(compute_thermal_conductivity(T=1000 + 273.15, Pr=0.7260, cp=1.184) - 7.868e-5) / 7.868e-5 < 1e-2
