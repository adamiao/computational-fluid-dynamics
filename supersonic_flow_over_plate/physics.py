from supersonic_flow_over_plate.constants import (
    TEMPERATURE_E,
    VISCOSITY_E,
    CP,
    PRANDTL,
)


def compute_viscosity(T, T_ref=TEMPERATURE_E, mu_ref=VISCOSITY_E):
    """
    Computation of the viscosity using Sutherland's law
    Args:
        T: temperature [K]
        T_ref: reference temperature [K]
        mu_ref: reference viscosity [kg / m^3]

    Returns: float representing the viscosity at node (i, j) [kg / (m * s)]
    """

    return mu_ref * (T / T_ref) ** 1.5 * ((T_ref + 110.0) / (T + 110.0))


def compute_thermal_conductivity(T, T_ref=TEMPERATURE_E, mu_ref=VISCOSITY_E, cp=CP, Pr=PRANDTL):
    """
    Computation of the thermal conductivity
    Args:
        T: temperature [K]
        T_ref: reference temperature [K]
        mu_ref: reference viscosity [kg / (m * s)]
        cp: float representing the specific heat at constant pressure [kJ / (kg * K)]
        Pr: float representing the flow's Prandtl number

    Returns: float representing the thermal conductivity [kW / (m * K)]
    """

    mu = compute_viscosity(T, T_ref, mu_ref)
    return mu * cp / Pr
