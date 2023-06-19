import numpy as np
from supersonic_flow_over_plate.constants import (
    N_X,
    N_Y,
    U_E,
    V_E,
    TEMPERATURE_E,
    TEMPERATURE_WALL,
    PRESSURE_E,
    CV,
    R,
)


def initial_conditions(x_speed=U_E, y_speed=V_E, temperature=TEMPERATURE_E, pressure=PRESSURE_E,
                       temperature_wall=TEMPERATURE_WALL):
    """
    Computation of the initial conditions for the fluid flow over the flat plate.
    Args:
        x_speed: float representing the x-direction speed at the entry of the computational grid
        y_speed: float representing the y-direction speed at the entry of the computational grid
        temperature: float representing the temperature at the entry of the computational grid
        pressure: float representing the pressure at the entry of the computational grid
        temperature_wall: float representing the wall temperature of flat plate

    Returns: a tuple of numpy arrays representing, respectfully:
    x-speed, y-speed, temperature, density, pressure, specific energy
    """

    # Initialize velocity and apply boundary condition
    u, v = np.full(shape=(N_Y, N_X), fill_value=x_speed), np.full(shape=(N_Y, N_X), fill_value=y_speed)
    u[-1, :] = 0  # no-slip condition
    v[-1, :] = 0  # no-slip condition

    # Initialize temperature and pressure
    T, p = np.full(shape=(N_Y, N_X), fill_value=temperature), np.full(shape=(N_Y, N_X), fill_value=pressure)
    T[-1, 1:] = temperature_wall  # except for the leading edge, the plate has a wall temperature

    # Calculate the density from the other flow variables
    rho = p / (R * T)

    # Calculate the specific energy of the flow
    e = CV * T

    return u, v, T, rho, p, e
