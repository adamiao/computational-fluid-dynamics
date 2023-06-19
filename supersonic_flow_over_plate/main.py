import numpy as np
import matplotlib.pyplot as plt
from supersonic_flow_over_plate.initial_conditions import initial_conditions
from supersonic_flow_over_plate.maccormack_method import maccormack
from supersonic_flow_over_plate.constants import (
    N_X,
    N_Y,
    DELTA_X,
    DELTA_Y,
    ITERATIONS_CONSTANT_TEMPERATURE,
    ITERATIONS_ADIABATIC,
    PRESSURE_E,
    TEMPERATURE_E,
    U_E,
    REYNOLDS,
)


if __name__ == '__main__':
    # Initialize the primitive and the derived variables for the constant temperature and adiabatic boundary conditions
    u_ct, v_ct, T_ct, rho_ct, p_ct, e_ct = initial_conditions()
    u_ad, v_ad, T_ad, rho_ad, p_ad, e_ad = initial_conditions()
    total_time_ct, total_time_ad = 0.0, 0.0
    for _ in range(ITERATIONS_CONSTANT_TEMPERATURE):
        u_ct, v_ct, T_ct, rho_ct, p_ct, e_ct, dt_ct = maccormack(u_ct, v_ct, T_ct, rho_ct, p_ct,
                                                                 boundary_type='constant_temperature')
        total_time_ct += dt_ct

    for _ in range(ITERATIONS_ADIABATIC):
        u_ad, v_ad, T_ad, rho_ad, p_ad, e_ad, dt_ad = maccormack(u_ad, v_ad, T_ad, rho_ad, p_ad,
                                                                 boundary_type='adiabatic')
        total_time_ad += dt_ad

    # Print flow times for both boundary condition types
    print(f'Total flow time for constant temperature boundary condition:\t{total_time_ct:.3e} seconds')
    print(f'Total flow time for adiabatic boundary condition:\t\t\t\t{total_time_ad:.3e} seconds')

    # Calculate the dimensions of the plate horizontally and vertically
    x_positions = np.array([i * DELTA_X for i in range(N_X)])
    y_positions = np.array([i * DELTA_Y for i in range(N_Y)])

    # Plot the normalized pressure versus distance along the flat plate (at the surface)
    fig = plt.figure(figsize=(10, 8))
    pressure_normalized_ct = p_ct[-1, :] / PRESSURE_E
    pressure_normalized_ad = p_ad[-1, :] / PRESSURE_E
    plt.plot(x_positions, pressure_normalized_ct, c='red')
    plt.plot(x_positions, pressure_normalized_ad, c='blue')
    plt.scatter(x_positions, pressure_normalized_ct, c='red')
    plt.scatter(x_positions, pressure_normalized_ad, c='blue')
    x_major_ticks = np.arange(0, x_positions[-1] * 1.02, 1e-6)
    x_minor_ticks = np.arange(0, x_positions[-1] * 1.02, 5e-7)
    y_major_ticks = np.arange(0, 7, 1)
    y_minor_ticks = np.arange(0, 7, 0.5)
    plt.xlabel(r'Distance along the flat plate (m)')
    plt.ylabel(r'$\frac{P}{P_{\infty}}$', rotation=0, fontsize=20)
    plt.gca().yaxis.set_label_coords(-0.075, 0.5)
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Normalized Pressure at the Surface')
    plt.legend(['Constant Temperature', 'Adiabatic'])
    # plt.savefig('normalized_pressure_surface.png')
    plt.show()

    # Plot the normalized pressure versus height above flat plate (at the trailing edge)
    fig = plt.figure(figsize=(10, 8))
    y_normalized = (y_positions * REYNOLDS ** 0.5) / x_positions[-1]
    pressure_normalized_ct = p_ct[::-1, -1] / PRESSURE_E
    pressure_normalized_ad = p_ad[::-1, -1] / PRESSURE_E
    plt.plot(pressure_normalized_ct, y_normalized, c='red')
    plt.plot(pressure_normalized_ad, y_normalized, c='blue')
    plt.scatter(pressure_normalized_ct, y_normalized, c='red')
    plt.scatter(pressure_normalized_ad, y_normalized, c='blue')
    x_major_ticks = np.arange(0.8, 2.6, 0.10)
    x_minor_ticks = np.arange(0.8, 2.6, 0.05)
    y_major_ticks = np.arange(0, 26, 5)
    y_minor_ticks = np.arange(0, 26, 2.5)
    plt.xlabel(r'Normalized Pressure ($P/P_{\infty}$)')
    plt.ylabel(r'$\frac{y}{x}\;\sqrt{Re_x}$', rotation=0, fontsize=15)
    plt.gca().yaxis.set_label_coords(-0.075, 0.5)
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Normalized Pressure at the Trailing Edge of the Flat Plate')
    plt.legend(['Constant Temperature', 'Adiabatic'])
    # plt.savefig('normalized_pressure_trailing_edge.png')
    plt.show()

    # Plot the normalized temperature versus height above flat plate (at the trailing edge)
    fig = plt.figure(figsize=(10, 8))
    y_normalized = (y_positions * REYNOLDS ** 0.5) / x_positions[-1]
    temperature_normalized_ct = T_ct[::-1, -1] / TEMPERATURE_E
    temperature_normalized_ad = T_ad[::-1, -1] / TEMPERATURE_E
    plt.plot(temperature_normalized_ct, y_normalized, c='red')
    plt.plot(temperature_normalized_ad, y_normalized, c='blue')
    plt.scatter(temperature_normalized_ct, y_normalized, c='red')
    plt.scatter(temperature_normalized_ad, y_normalized, c='blue')
    x_major_ticks = np.arange(0.8, 3.6, 0.20)
    x_minor_ticks = np.arange(0.8, 3.6, 0.10)
    y_major_ticks = np.arange(0, 26, 5)
    y_minor_ticks = np.arange(0, 26, 2.5)
    plt.xlabel(r'Normalized Temperature ($T/T_{\infty}$)')
    plt.ylabel(r'$\frac{y}{x}\;\sqrt{Re_x}$', rotation=0, fontsize=15)
    plt.gca().yaxis.set_label_coords(-0.08, 0.5)
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Normalized Temperature at the Trailing Edge of the Flat Plate')
    plt.legend(['Constant Temperature', 'Adiabatic'])
    # plt.savefig('normalized_temperature_trailing_edge.png')
    plt.show()

    # Plot the normalized x-speed versus height above flat plate (at the trailing edge)
    fig = plt.figure(figsize=(10, 8))
    y_normalized = (y_positions * REYNOLDS ** 0.5) / x_positions[-1]
    x_speed_normalized_ct = u_ct[::-1, -1] / U_E
    x_speed_normalized_ad = u_ad[::-1, -1] / U_E
    plt.plot(x_speed_normalized_ct, y_normalized, c='red')
    plt.plot(x_speed_normalized_ad, y_normalized, c='blue')
    plt.scatter(x_speed_normalized_ct, y_normalized, c='red')
    plt.scatter(x_speed_normalized_ad, y_normalized, c='blue')
    x_major_ticks = np.arange(0, 1.05, 0.10)
    x_minor_ticks = np.arange(0, 1.05, 0.05)
    y_major_ticks = np.arange(0, 26, 5)
    y_minor_ticks = np.arange(0, 26, 2.5)
    plt.xlabel(r'Normalized Speed in the x-direction ($U/U_{\infty}$)')
    plt.ylabel(r'$\frac{y}{x}\;\sqrt{Re_x}$', rotation=0, fontsize=15)
    plt.gca().yaxis.set_label_coords(-0.075, 0.5)
    plt.xticks(x_major_ticks)
    plt.xticks(x_minor_ticks, minor=True)
    plt.yticks(y_major_ticks)
    plt.yticks(y_minor_ticks, minor=True)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.title('Normalized Speed (x-direction) at the Trailing Edge of the Flat Plate')
    plt.legend(['Constant Temperature', 'Adiabatic'])
    # plt.savefig('normalized_x_speed_trailing_edge.png')
    plt.show()

    # Plot the absolute speed profile for the constant temperature boundary condition case flow
    x_p = np.array([x_positions for _ in range(N_Y)])
    y_p = np.array([y_positions[::-1] for _ in range(N_X)]).transpose()
    speed_ct = (u_ct ** 2 + v_ct ** 2) ** 0.5
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    img = ax1.contourf(x_p, y_p, speed_ct, cmap='Spectral', alpha=1.0)
    cbar = fig.colorbar(img, orientation='vertical')
    cbar.set_label(r'Speed  ($m/s$)')
    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$y$ (m)')
    ax1.set_ylim(0, 8e-6)
    plt.title('Speed Profile for the Constant Temperature Boundary Condition')
    # plt.savefig('speed_profile_constant_temperature.png')
    plt.show()

    # Plot the absolute speed profile for the adiabatic boundary condition case flow
    speed_ad = (u_ad ** 2 + v_ad ** 2) ** 0.5
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    img = ax1.contourf(x_p, y_p, speed_ad, cmap='Spectral', alpha=1.0)
    cbar = fig.colorbar(img, orientation='vertical')
    cbar.set_label(r'Speed  ($m/s$)')
    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$y$ (m)')
    ax1.set_ylim(0, 8e-6)
    plt.title('Speed Profile for the Adiabatic Boundary Condition')
    # plt.savefig('speed_profile_adiabatic.png')
    plt.show()
