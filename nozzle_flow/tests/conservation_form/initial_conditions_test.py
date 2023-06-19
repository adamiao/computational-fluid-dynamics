from nozzle_flow.conservation_form.initial_and_boundary_conditions import initial_conditions


def test_initial_conditions():
    U1, U2, U3 = initial_conditions()

    # Test U1
    assert abs(U1[0] - 5.950) < 1e-3
    assert abs(U1[1] - 5.312) < 1e-3
    assert abs(U1[-2] - 0.483) < 1e-3
    assert abs(U1[-1] - 0.310) < 1e-3

    # Test U2
    assert abs(U2[0] - 0.59) < 1e-3
    assert abs(U2[1] - 0.59) < 1e-3
    assert abs(U2[-2] - 0.59) < 1e-3
    assert abs(U2[-1] - 0.59) < 1e-3

    # Test U3
    assert abs(U3[0] - 14.916) < 1e-3
    assert abs(U3[1] - 13.326) < 1e-3
    assert abs(U3[-2] - 0.917) < 1e-3
    assert abs(U3[-1] - 1.023) < 1e-3
