import matplotlib.pyplot as plt
from expansion_wave.initial_and_boundary_conditions import initial_conditions
from expansion_wave.predictor_corrector_step import predictor_corrector
from expansion_wave.variable_conversion import decode_primitive_variables
from expansion_wave.geometry import coordinate_systems
from expansion_wave.constants import GAMMA, R


if __name__ == '__main__':
    # Create the initial conditions and run the algorithm
    F1, F2, F3, F4, X = initial_conditions()
    for j in range(X.size - 1):
        predictor_corrector(F1, F2, F3, F4, X, j)

    # Decode the flow variables from the converged solution and calculate Mach number
    rho, u, v, p, T = decode_primitive_variables(F1, F2, F3, F4)
    a = (GAMMA * R * T) ** 0.5
    M = (u ** 2 + v ** 2) ** 0.5 / a

    # Create the visualization of the solution for the density as an example
    x, y = coordinate_systems(X)
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    img = ax1.contourf(x, y, rho, cmap='Spectral', alpha=1.0)
    cbar = fig.colorbar(img, orientation='vertical')
    cbar.set_label(r'$\rho$  ($kg/m^3$)')
    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$y$ (m)')

    ax2.scatter(u[:, 50], y[:, 50], s=10, c='black')
    ax2.set_xlabel(r'$u$ ($m/s$) at $x$ = 43.32 m')
    ax2.set_xlim(400, 850)

    plt.savefig('prandtl_meyer_expansion_wave.png')
    plt.show()

    # # (ORIGINAL PLOT) Create the visualization of the solution for the density as an example
    # x, y = coordinate_systems(X)
    # plt.contourf(x, y, rho, cmap='Spectral', alpha=1.0)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\rho$  ($kg/m^3$)')
    #
    # plt.title('Density Field')
    # plt.xlabel(r'$x$ (m)')
    # plt.ylabel(r'$y$ (m)')
    #
    # plt.tight_layout()
    #
    # plt.show()
