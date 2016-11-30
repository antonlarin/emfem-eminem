#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fem_2d import fem_2d, Problem2D
from mesh import load_msh

# equation setup
mesh = load_msh('parallel-plate-waveguide-discontinuity.msh')

k0 = 20 * math.pi
H0 = 1
eps_r = lambda x, y: 4 - 10j if (y <= 0.0175 and 0.1 <= x and x <= 0.15) else 1
mu_r = lambda x, y: 1

alpha_x = lambda x, y: 1 / eps_r(x, y)
alpha_y = lambda x, y: 1 / eps_r(x, y)
beta = lambda x, y: -k0**2 * mu_r(x, y)
f = lambda x, y: 0

# boundary condition setup
# - homogenous Neumann conditions on top and bottom, 3rd kind on left and right
def close_to(x, y):
    return abs(x - y) <= 1e-8

def common_gamma(x, y):
    if close_to(y, 0.035) or close_to(y, 0): # top or bottom
        return 0
    elif close_to(x, 0): # left
        return 1j * k0 / eps_r(0, y)
    elif close_to(x, 0.25): # right
        return 1j * k0 / eps_r(0.25, y)
    else: # inside, value shouldn't be used
        return None

def common_q(x, y):
    if close_to(y, 0.035) or close_to(y, 0): # top or bottom
        return 0
    elif close_to(x, 0): # left
        return 2j * k0 * H0 / eps_r(0, y)
    elif close_to(x, 0.25): # right
        return 0
    else: # inside, value shouldn't be used
        return None

gamma = lambda x, y: common_gamma(x, y)
q = lambda x, y: common_q(x, y)

# - No dirichlet boundary
p = lambda x, y: None

problem = Problem2D(alpha_x, alpha_y, beta, f, gamma, q, p)

def main():
    phis = fem_2d(problem, mesh, dtype='complex')
    phis_real = lambda x, y: phis(x, y).real
    phis_imag = lambda x, y: phis(x, y).imag

    # plot results
    samples_x = 200
    samples_y = samples_x * 7 // 50
    xs = np.linspace(0., .25, samples_x)
    ys = np.linspace(0., .035, samples_y)
    plot_grid = np.meshgrid(xs, ys)

    phis_real_vec = np.vectorize(phis_real)
    phis_imag_vec = np.vectorize(phis_imag)
    phis_real_grid = phis_real_vec(*plot_grid)
    phis_imag_grid = phis_imag_vec(*plot_grid)

    plt.gcf().set_size_inches(10, 6)
    plt.subplot(211)
    plt.contourf(plot_grid[0], plot_grid[1], phis_real_grid, 15,
            interpolation='nearest', cmap='viridis',
            vmin=abs(phis_real_grid).min(), vmax=abs(phis_real_grid).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], ys[0], ys[-1]])
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathfrak{Re}(H_z)$')
    plt.contour(plot_grid[0], plot_grid[1], phis_real_grid, 15,
            interpolation='nearest', colors='k',
            vmin=abs(phis_real_grid).min(), vmax=abs(phis_real_grid).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], ys[0], ys[-1]])
    plt.gca().add_patch(Rectangle((0.1, 0), 0.05, 0.0175, fill=False, ec='cyan'))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.subplot(212)
    plt.contourf(plot_grid[0], plot_grid[1], phis_imag_grid, 15,
            interpolation='nearest', cmap='viridis',
            vmin=abs(phis_imag_grid).min(), vmax=abs(phis_imag_grid).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], ys[0], ys[-1]])
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathfrak{Im}(H_z)$')
    plt.contour(plot_grid[0], plot_grid[1], phis_imag_grid, 15,
            interpolation='nearest', colors='k',
            vmin=abs(phis_imag_grid).min(), vmax=abs(phis_imag_grid).max(),
            aspect='auto', origin='lower',
            extent=[xs[0], xs[-1], ys[0], ys[-1]])
    plt.gca().add_patch(Rectangle((0.1, 0), 0.05, 0.0175, fill=False, ec='cyan'))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.subplots_adjust(right=0.99)
    plt.savefig('parallel-plate-waveguide-discontinuity-c.png', dpi=120)


if __name__ == '__main__':
    main()

