from expansion_wave.variable_conversion import decode_primitive_variables, calculate_derived_variables


def test_decode_primitive_variables():
    F1, F2, F3, F4 = 0.728e3, 0.590e6, -0.36e5, 0.375e9
    rho, u, v, p, T = decode_primitive_variables(F1, F2, F3, F4)

    assert abs(rho - 1.038) < 1e-3
    assert abs(u - 701.236) < 1e-3
    assert abs(v - -49.451) < 1e-3
    assert abs(p - 79500.251) < 1e-3
    assert abs(T - 266.774) < 1e-3


def test_calculate_derived_variables():
    F1, F2, F3, F4 = 0.721e3, 0.585e6, -0.388e5, 0.372e9
    G1, G2, G3, G4 = calculate_derived_variables(F1, F2, F3, F4)

    assert abs(G1 + 55.235) < 1e-3
    assert abs(G2 + 38800.0) < 1e-3
    assert abs(G3 - 81499.919) < 1e-3
    assert abs(G4 + 28498290.55) < 1e-3
