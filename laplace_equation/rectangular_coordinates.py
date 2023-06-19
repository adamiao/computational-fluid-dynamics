import numpy as np
import matplotlib.pyplot as plt


# initial parameters
grid_size = 100
input_square_size = 5
input_square_location = (grid_size - input_square_size) // 2
phi_boundary_value = [50, 50, 50, 50]
input_dx, input_dy, n_iterations = 1, 1, 2_000

# initialize solution with boundary conditions
phi = np.zeros((grid_size, grid_size))
phi[0, :] = phi_boundary_value[0]
phi[:, 0] = phi_boundary_value[1]
phi[grid_size - 1, :] = phi_boundary_value[2]
phi[:, grid_size - 1] = phi_boundary_value[3]

# phi[square_location:square_location + square_size, square_location:square_location + square_size] = -999


# method that performs the 'relaxation' technique to solve Laplace's equation in rectangular coordinates
def relaxation(grid, dx, dy, square_location, square_size):
    m, n = grid.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if (square_location < i < square_location + square_size - 1) and \
                    (square_location < j < square_location + square_size - 1):
                continue
            cnt = (dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))
            term_0 = 1 / (dx ** 2) * (phi[i + 1, j] + phi[i - 1, j])
            term_1 = 1 / (dy ** 2) * (phi[i, j + 1] + phi[i, j - 1])
            phi[i, j] = cnt * (term_0 + term_1)
    return grid


if __name__ == '__main__':

    for _ in range(n_iterations):
        phi = relaxation(phi, input_dx, input_dy, input_square_location, input_square_size)

    # display 2d array with a plot
    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(phi)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
