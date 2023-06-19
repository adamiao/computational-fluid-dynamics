import numpy as np
from expansion_wave.constants import THETA, X_EXPANSION, H, N_X, N_Y, DELTA_ETA


def ys(x):
    tangent_theta = np.tan(np.radians(THETA))
    return -(x - X_EXPANSION) * tangent_theta if x > X_EXPANSION else 0


def height(x):
    tangent_theta = np.tan(np.radians(THETA))
    return H + (x - X_EXPANSION) * tangent_theta if x > X_EXPANSION else H


def xi_coordinate(x, y):
    return x


def eta_coordinate(x, y):
    tangent_theta = np.tan(np.radians(THETA))
    constant = tangent_theta * (x - X_EXPANSION)
    return (y + constant) / (H + constant) if x > X_EXPANSION else y / H


def derivative_eta_x(xi, eta):
    tangent_theta = np.tan(np.radians(THETA))
    return (1 - eta) * tangent_theta / height(xi) if xi > X_EXPANSION else 0.0


def coordinate_systems(XI):
    # The x-coordinate system is the same as the 'XI' coordinate system
    x_coordinates = XI

    # Initialize the y-coordinate system
    y_coordinates = np.zeros(shape=(N_Y, N_X))
    for i in range(N_Y):
        eta = i * DELTA_ETA
        for j in range(N_X):
            y_coordinates[i, j] = eta * height(x_coordinates[j]) + ys(x_coordinates[j])

    # Here we will update the 'x-coordinates' so that we have it with the same shape as the 'y-coordinates'. This is
    # done for plotting purposes.
    x_coordinates = np.array([x_coordinates for _ in range(N_Y)])

    return x_coordinates, y_coordinates
