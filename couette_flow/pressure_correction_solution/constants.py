# GEOMETRY
D = 1e-2  # distance between the plates [feet]
L = 5e-1  # length of the computational space [feet]
NX_P = 21  # number of nodes in the x-coordinate system for the pressure grid
NY_P = 11  # number of nodes in the y-coordinate system for the pressure grid

# GEOMETRY DERIVED CONSTANTS
NX_U = NX_P + 1  # number of nodes in the x-coordinate system for the x-velocity grid
NY_U = NY_P  # number of nodes in the y-coordinate system for the x-velocity grid
NX_V = NX_P + 1  # number of nodes in the x-coordinate system for the x-velocity grid
NY_V = NY_P  # number of nodes in the y-coordinate system for the x-velocity grid
DELTA_X = L / (NX_P - 1)  # space variation in the x-coordinate system
DELTA_Y = D / (NY_P - 1)  # space variation in the y-coordinate system

# INITIAL CONDITIONS AND PHYSICS
U_E = 1  # initial speed of the top plate [feet / s]
V_E = 5e-1  # initial y-coordinate speed in the middle of the computational grid [feet / s]
DENSITY = 2.377e-3  # slug / ft ^ 3
VISCOSITY = 3.737e-7  # viscosity of air [slug / (ft * s)]
REYNOLDS = DENSITY * U_E * D / VISCOSITY  # Reynolds number
DELTA_T = 1e-3  # time variation

# NUMERICAL PARAMETERS
RELAXATION_CONSTANT = 1e-1  # under-relaxation factor
RELAXATION_ITERATIONS = 500  # number of iterations that will be performed for the relaxation technique
ITERATIONS = 301  # number of iterations that the whole code will run for
