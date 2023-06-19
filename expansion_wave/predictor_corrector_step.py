import numpy as np
from expansion_wave.physics import mach_angle
from expansion_wave.initial_and_boundary_conditions import boundary_conditions
from expansion_wave.variable_conversion import decode_primitive_variables, calculate_derived_variables
from expansion_wave.geometry import derivative_eta_x, height
from expansion_wave.constants import DELTA_ETA, N_Y, GAMMA, X_EXPANSION, THETA, R, CFL, C_Y


def space_step(F1, F2, F3, F4, xi):
    # Determine flow variables based on the derived variables
    rho, u, v, p, T = decode_primitive_variables(F1, F2, F3, F4)
    a = (GAMMA * R * T) ** 0.5
    M = (u ** 2 + v ** 2) ** 0.5 / a

    # Determine the maximum denominator for the downstream marching step size as explained in pages 395-397 of
    # John Anderson's CFD book
    theta = 0.0 if xi <= X_EXPANSION else np.radians(-THETA)
    mu_angles = mach_angle(M, in_degrees=False)
    denominator_minus = np.max(np.abs(np.tan(mu_angles - theta)))
    denominator_plus = np.max(np.abs(np.tan(mu_angles + theta)))
    denominator = max(denominator_minus, denominator_plus)

    # Compute the downstream marching step size
    DELTA_Y = height(xi) / (N_Y - 1)
    dxi = CFL * DELTA_Y / denominator

    return dxi


def artificial_viscosity(F1, F2, F3, F4):
    _, _, _, p, _ = decode_primitive_variables(F1, F2, F3, F4)
    SF1, SF2, SF3, SF4 = np.zeros(p.size), np.zeros(p.size), np.zeros(p.size), np.zeros(p.size)
    for i in range(1, p.size - 1):
        term_0 = C_Y * np.abs(p[i + 1] - 2 * p[i] + p[i - 1]) / (p[i + 1] + 2 * p[i] + p[i - 1])
        SF1[i] = term_0 * (F1[i + 1] - 2 * F1[i] + F1[i - 1])
        SF2[i] = term_0 * (F2[i + 1] - 2 * F2[i] + F2[i - 1])
        SF3[i] = term_0 * (F3[i + 1] - 2 * F3[i] + F3[i - 1])
        SF4[i] = term_0 * (F4[i + 1] - 2 * F4[i] + F4[i - 1])
    return SF1, SF2, SF3, SF4


def forward_difference(F1, F2, F3, F4, G1, G2, G3, G4, xi, i):
    deta_dx = derivative_eta_x(xi, i * DELTA_ETA)
    dF1_dxi = deta_dx * (F1[i] - F1[i + 1]) / DELTA_ETA + 1 / height(xi) * (G1[i] - G1[i + 1]) / DELTA_ETA
    dF2_dxi = deta_dx * (F2[i] - F2[i + 1]) / DELTA_ETA + 1 / height(xi) * (G2[i] - G2[i + 1]) / DELTA_ETA
    dF3_dxi = deta_dx * (F3[i] - F3[i + 1]) / DELTA_ETA + 1 / height(xi) * (G3[i] - G3[i + 1]) / DELTA_ETA
    dF4_dxi = deta_dx * (F4[i] - F4[i + 1]) / DELTA_ETA + 1 / height(xi) * (G4[i] - G4[i + 1]) / DELTA_ETA
    return dF1_dxi, dF2_dxi, dF3_dxi, dF4_dxi


def backward_difference(F1, F2, F3, F4, G1, G2, G3, G4, xi, i):
    deta_dx = derivative_eta_x(xi, i * DELTA_ETA)
    dF1_dxi = deta_dx * (F1[i - 1] - F1[i]) / DELTA_ETA + 1 / height(xi) * (G1[i - 1] - G1[i]) / DELTA_ETA
    dF2_dxi = deta_dx * (F2[i - 1] - F2[i]) / DELTA_ETA + 1 / height(xi) * (G2[i - 1] - G2[i]) / DELTA_ETA
    dF3_dxi = deta_dx * (F3[i - 1] - F3[i]) / DELTA_ETA + 1 / height(xi) * (G3[i - 1] - G3[i]) / DELTA_ETA
    dF4_dxi = deta_dx * (F4[i - 1] - F4[i]) / DELTA_ETA + 1 / height(xi) * (G4[i - 1] - G4[i]) / DELTA_ETA
    return dF1_dxi, dF2_dxi, dF3_dxi, dF4_dxi


def predictor_corrector(F1, F2, F3, F4, X, j):
    # # # # # # # # # # #
    #                   #
    #   PREDICTOR STEP  #
    #                   #
    # # # # # # # # # # #

    # Initialize the numpy arrays to keep track of the derivatives
    dF1_dxi, dF2_dxi, dF3_dxi, dF4_dxi = np.zeros(N_Y), np.zeros(N_Y), np.zeros(N_Y), np.zeros(N_Y)

    # Calculates the derived variables that will be needed for this calculation at each 'j' node location
    G1, G2, G3, G4 = calculate_derived_variables(F1[:, j], F2[:, j], F3[:, j], F4[:, j])

    # Calculate the artificial viscosity that will be used to stabilize numerical calculation
    SF1, SF2, SF3, SF4 = artificial_viscosity(F1[:, j], F2[:, j], F3[:, j], F4[:, j])

    # Predictor step
    for i in range(N_Y):
        if i != N_Y - 1:
            dF1_dxi[i], dF2_dxi[i], dF3_dxi[i], dF4_dxi[i] = forward_difference(
                F1[:, j], F2[:, j], F3[:, j], F4[:, j], G1, G2, G3, G4, X[j], i
            )
        else:
            dF1_dxi[i], dF2_dxi[i], dF3_dxi[i], dF4_dxi[i] = backward_difference(
                F1[:, j], F2[:, j], F3[:, j], F4[:, j], G1, G2, G3, G4, X[j], i
            )

    # Predict the space marching variation in the 'xi_coordinate' system
    delta_xi = space_step(F1[:, j], F2[:, j], F3[:, j], F4[:, j], X[j])

    # Calculate the predicted values for F1, F2, F3, F4 (note that the artificial viscosity is used here)
    pred_F1 = F1[:, j] + dF1_dxi * delta_xi + SF1
    pred_F2 = F2[:, j] + dF2_dxi * delta_xi + SF2
    pred_F3 = F3[:, j] + dF3_dxi * delta_xi + SF3
    pred_F4 = F4[:, j] + dF4_dxi * delta_xi + SF4

    # Calculate the predicted values for the remaining derived variables G1, G2, G3, G4
    pred_G1, pred_G2, pred_G3, pred_G4 = calculate_derived_variables(pred_F1, pred_F2, pred_F3, pred_F4)

    # Calculate the predicted artificial viscosity
    pred_SF1, pred_SF2, pred_SF3, pred_SF4 = artificial_viscosity(pred_F1, pred_F2, pred_F3, pred_F4)

    # # # # # # # # # # #
    #                   #
    #   CORRECTOR STEP  #
    #                   #
    # # # # # # # # # # #

    # Initialize the numpy arrays to keep track of the predicted derivatives for derived variables
    pred_dF1_dxi, pred_dF2_dxi, pred_dF3_dxi, pred_dF4_dxi = np.zeros(N_Y), np.zeros(N_Y), np.zeros(N_Y), np.zeros(N_Y)

    # Corrector step
    for i in range(N_Y):
        if i == 0:
            pred_dF1_dxi[i], pred_dF2_dxi[i], pred_dF3_dxi[i], pred_dF4_dxi[i] = forward_difference(
                pred_F1, pred_F2, pred_F3, pred_F4, pred_G1, pred_G2, pred_G3, pred_G4, X[j], i
            )
        else:
            pred_dF1_dxi[i], pred_dF2_dxi[i], pred_dF3_dxi[i], pred_dF4_dxi[i] = backward_difference(
                pred_F1, pred_F2, pred_F3, pred_F4, pred_G1, pred_G2, pred_G3, pred_G4, X[j], i
            )

    # Calculate the average derivative of the derived variables
    dF1_avg = 0.5 * (dF1_dxi + pred_dF1_dxi)
    dF2_avg = 0.5 * (dF2_dxi + pred_dF2_dxi)
    dF3_avg = 0.5 * (dF3_dxi + pred_dF3_dxi)
    dF4_avg = 0.5 * (dF4_dxi + pred_dF4_dxi)

    # Calculate the next values (in space) for the derived variables
    F1[:, j + 1] = F1[:, j] + dF1_avg * delta_xi + pred_SF1
    F2[:, j + 1] = F2[:, j] + dF2_avg * delta_xi + pred_SF2
    F3[:, j + 1] = F3[:, j] + dF3_avg * delta_xi + pred_SF3
    F4[:, j + 1] = F4[:, j] + dF4_avg * delta_xi + pred_SF4

    # Update location of the calculation
    X[j + 1] = X[j] + delta_xi

    # Apply boundary conditions (velocity is parallel to the wall)
    boundary_conditions(F1[:, j + 1], F2[:, j + 1], F3[:, j + 1], F4[:, j + 1], X[j + 1])

    return
