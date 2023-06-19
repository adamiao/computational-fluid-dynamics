# PHYSICS
REYNOLDS = 5_000  # Reynolds number

# GEOMETRY
N = 21  # number of nodes in the y-coordinate system

# INITIAL CONDITIONS
U_E = 1  # initial speed of the top plate

# NUMERICAL PARAMETERS
DELTA_T = 12.5  # time-marching step variation
DELTA_Y = 1 / (N - 1)  # vertical variation between nodes
E = 1  # computational constant equal to DELTA_T / (REYNOLDS * DELTA_Y ** 2)
