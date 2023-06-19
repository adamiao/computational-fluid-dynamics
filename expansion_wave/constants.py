# PHYSICS
GAMMA = 1.4  # specific heat ratio for air
R = 287.05  # specific gas constant for air [J * kg^-1 * K^-1]

# GEOMETRY
L = 65.0  # x-coordinate system length of the physical plane [meters]
H = 40  # y-coordinate system height of the upper boundary [meters]
N_X = 81  # number of nodes in the x/xi-coordinate systems
N_Y = 41  # number of nodes in the y/eta-coordinate systems
X_EXPANSION = 10  # location of the expansion corner in the x-coordinate system [meters]
THETA = 5.353  # expansion corner angle [degrees]
DELTA_ETA = 1.0 / (N_Y - 1)

# INITIAL CONDITIONS
MACH_0 = 2  # Mach number
T_0 = 286.1  # [K]
P_0 = 1.01e5  # [N * m^-2]
RHO_0 = 1.23  # [kg * m^-3]

# NUMERICAL PARAMETERS
CFL = 0.5  # Courant-Friedrichs-Lowry (CFL) criterion constant
C_Y = 0.6  # Artificial viscosity constant
