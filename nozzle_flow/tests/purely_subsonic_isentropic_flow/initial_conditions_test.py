from nozzle_flow.purely_subsonic_isentropic_flow.initial_and_boundary_conditions import initial_conditions


def test_initial_conditions():
    rho, u, T, p, M = initial_conditions()

    # Test density
    assert abs(rho[0] - 1.0000) < 1e-3
    assert abs(rho[1] - 0.9977) < 1e-3
    assert abs(rho[-2] - 0.9333) < 1e-3
    assert abs(rho[-1] - 0.931) < 1e-3

    # Test speed
    assert abs(u[0] - 0.050) < 1e-3
    assert abs(u[1] - 0.061) < 1e-3
    assert abs(u[-2] - 0.369) < 1e-3
    assert abs(u[-1] - 0.380) < 1e-3

    # Test temperature
    assert abs(T[0] - 1.0000) < 1e-3
    assert abs(T[1] - 0.999) < 1e-3
    assert abs(T[-2] - 0.973) < 1e-3
    assert abs(T[-1] - 0.972) < 1e-3

    # Test pressure
    assert abs(p[0] - 1.000) < 1e-3
    assert abs(p[1] - 0.997) < 1e-3
    assert abs(p[-2] - 0.908) < 1e-3
    assert abs(p[-1] - 0.905) < 1e-3

    # Test Mach number
    assert abs(M[0] - 0.050) < 1e-3
    assert abs(M[1] - 0.061) < 1e-3
    assert abs(M[-2] - 0.374) < 1e-3
    assert abs(M[-1] - 0.385) < 1e-3
