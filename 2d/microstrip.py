#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from fem_2d import fem_2d, Problem2D
from mesh import load_msh

# equation setup
mesh = load_msh('microstrip.msh')
eps_r = lambda x, y: 1 if y > 0.15 else 2.5
alpha_x = lambda x, y: eps_r(x, y)
alpha_y = lambda x, y: eps_r(x, y)
beta = lambda x, y: 0
f = lambda x, y: 0 # zero charge density

# boundary condition setup
# - homogenous Neumann conditions on the axis of symmetry
gamma = lambda x, y: 0
q = lambda x, y: 0

# - 1V on inner shield shell, 0V on outer
p = lambda x, y: 1 if (x < 0.3 and 0.1 < y and y < 0.2) else 0

problem = Problem2D(alpha_x, alpha_y, beta, f, gamma, q, p)

def main():
    phis = fem_2d(problem, mesh)

    # plot results
    samples = 100
    xs = np.linspace(0., 1., samples)
    plot_grid = np.meshgrid(xs, xs)

    phis_vec = np.vectorize(phis)
    phi = phis_vec(*plot_grid)

    plt.contourf(plot_grid[0], plot_grid[1], phi, 15, interpolation='nearest',
            cmap='viridis', vmin=abs(phi).min(), vmax=abs(phi).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], xs[0], xs[-1]])
    cbar = plt.colorbar()
    plt.contour(plot_grid[0], plot_grid[1], phi, 15, interpolation='nearest',
            colors='k', vmin=abs(phi).min(), vmax=abs(phi).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], xs[0], xs[-1]])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Equipotential lines')

    plt.savefig('microstrip.png', dpi=120)


if __name__ == '__main__':
    main()

