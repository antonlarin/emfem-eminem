#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from fem_2d import fem_2d, Problem2D
from mesh import load_msh

# equation setup
mesh = load_msh('coaxial-waveguides-join.msh')
eps_r = lambda x, y: 2.5
alpha_x = lambda x, y: eps_r(x, y) * y
alpha_y = lambda x, y: eps_r(x, y) * y
beta = lambda x, y: 0
f = lambda x, y: 0 # zero charge density

# boundary condition setup
# - homogenous Neumann conditions on the axis of symmetry
gamma = lambda x, y: 0
q = lambda x, y: 0

# - 1V on inner shell, 0V on outer
p = lambda x, y: 1 if y < 0.5 else 0

problem = Problem2D(alpha_x, alpha_y, beta, f, gamma, q, p)

def main():
    phis = fem_2d(problem, mesh)

    # plot results
    plt.gcf().set_size_inches(8, 4)
    samples_z = 100
    samples_rho = 40
    zs = np.linspace(0., 1., samples_z)
    rhos = np.linspace(.2, .6, samples_rho)
    plot_grid = np.meshgrid(zs, rhos)

    phis_vec = np.vectorize(phis)
    phi = phis_vec(*plot_grid)

    # plot equipotential lines
    plt.contour(plot_grid[0], plot_grid[1], phi, colors='k',
            vmin=abs(phi).min(), vmax=abs(phi).max(), aspect='auto',
            origin='lower', extent=[zs[0], zs[-1], rhos[0], rhos[-1]])
    plt.contourf(plot_grid[0], plot_grid[1], phi, cmap='viridis',
            vmin=abs(phi).min(), vmax=abs(phi).max(), aspect='auto',
            origin='lower', extent=[zs[0], zs[-1], rhos[0], rhos[-1]])
    cbar = plt.colorbar()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\rho$')
    plt.title('Equipotential lines')

    plt.subplots_adjust(bottom=0.15)

    plt.savefig('coaxial-waveguides-join.png', dpi=120)


if __name__ == '__main__':
    main()

