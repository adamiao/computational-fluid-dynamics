# GEOMETRY
L = 1e-5  # length of the computational space [meter]
N_X = 70  # number of nodes in the x-coordinate
N_Y = 70  # number of nodes in the y-coordinate

# PHYSICS
R = 287.05  # specific gas constant [J / (kg * K)]
CP = 1005.0  # specific heat at constant pressure [J / (kg * K)]
CV = 718.0  # specific heat at constant volume [J / (kg * K)]
GAMMA = CP / CV  # ratio of specific heats
SPEED_OF_SOUND_E = 340.28  # speed of sound at the entry of the physical domain [meter / second]
VISCOSITY_E = 1.7894e-5  # viscosity of air [kg / (m * s)]
PRANDTL = 0.71  # Prandtl number

# INITIAL CONDITIONS
MACH_E = 4  # Mach number at the entry of the physical domain
U_E = MACH_E * SPEED_OF_SOUND_E  # x-coordinate speed at the entry of the physical domain [meter / second]
V_E = 0.0  # y-coordinate speed at the entry of the physical domain [meter / second]
PRESSURE_E = 101325.0  # pressure at the entry of the physical domain [N / m^2]
TEMPERATURE_E = 288.16  # temperature at the entry of the physical domain [K]
TEMPERATURE_WALL = TEMPERATURE_E  # temperature at the wall will be always equal to the free stream temperature [K]
DENSITY_E = PRESSURE_E / (R * TEMPERATURE_E)  # 1.23  # density at the entry of the physical domain [kg / m^3]
REYNOLDS = DENSITY_E * U_E * L / VISCOSITY_E  # Reynolds number

# NUMERICAL PARAMETERS
ITERATIONS_CONSTANT_TEMPERATURE = 5_000  # number of iterations that the whole code will run for
ITERATIONS_ADIABATIC = 7_000  # number of iterations that the whole code will run for
COURANT = 0.7  # Courant number

# GEOMETRY DERIVED CONSTANTS
DELTA_X = L / (N_X - 1)  # space variation in the x-coordinate system
DELTA_Y = 25 * L / (REYNOLDS ** 0.5 * (N_Y - 1))  # space variation in the y-coordinate system
