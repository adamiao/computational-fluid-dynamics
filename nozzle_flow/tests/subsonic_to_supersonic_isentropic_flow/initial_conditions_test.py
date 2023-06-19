from nozzle_flow.subsonic_to_supersonic_isentropic_flow.initial_and_boundary_conditions import initial_conditions


def test_initial_conditions():
    rho, u, T, p, M = initial_conditions()

    # Test density
    assert abs(rho[0] - 1.0000) < 1e-3
    assert abs(rho[1] - 0.9690) < 1e-3
    assert abs(rho[-2] - 0.088) < 1e-3
    assert abs(rho[-1] - 0.056) < 1e-3

    # Test speed
    assert abs(u[0] - 0.1000) < 1e-3
    assert abs(u[1] - 0.2070) < 1e-3
    assert abs(u[-2] - 1.870) < 1e-3
    assert abs(u[-1] - 1.864) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.0000) < 1e-3
    assert abs(T[1] - 0.9770) < 1e-3
    assert abs(T[-2] - 0.329) < 1e-3
    assert abs(T[-1] - 0.306) < 1e-3

    # Test pressure
    assert abs(p[0] - 1.0000) < 1e-3
    assert abs(p[1] - 0.9470) < 1e-3
    assert abs(p[-2] - 0.029) < 1e-3
    assert abs(p[-1] - 0.017) < 1e-3

    # Test Mach number
    assert abs(M[0] - 0.1000) < 1e-3
    assert abs(M[1] - 0.2090) < 1e-3
    assert abs(M[-2] - 3.261) < 1e-3
    assert abs(M[-1] - 3.370) < 1e-3
