#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import matplotlib.pyplot as plt

from fem_2d import fem_2d, Problem2D
from mesh import load_msh

# equation setup
mesh = load_msh('center-refined-rect.msh')
alpha_x = lambda x, y: -1
alpha_y = lambda x, y: -1
beta = lambda x, y: 0
f = lambda x, y: math.exp(-0.5 * ((x - 0.5)**2 + (y - 0.5)**2)) * (
        x**2 + y**2 - x - y - 1.5)

# boundary condition setup
gamma = lambda x, y: 0
q = lambda x, y: -0.5 * math.exp(-0.5 * ((x - 0.5)**2 + (y - 0.5)**2))
p = lambda x, y: math.exp(-0.5 * ((x - 0.5)**2 + (y - 0.5)**2))

problem = Problem2D(alpha_x, alpha_y, beta, f, gamma, q, p)

def main():
    phis = fem_2d(problem, mesh)

    # plot results
    samples = 50
    xs = np.linspace(0., 1., samples)
    plotgrid = np.meshgrid(xs, xs)

    phis_vec = np.vectorize(lambda x, y: phis(x, y))
    solution_vec = np.vectorize(lambda x, y:
            math.exp(-0.5 * ((x - 0.5)**2 + (y - 0.5)**2)))
    phi = phis_vec(*plotgrid).reshape(samples, samples)
    solution = solution_vec(*plotgrid).reshape(samples, samples)

    plt.gcf().set_size_inches(10, 4)
    plt.subplot(121)
    plt.imshow(phi, interpolation='nearest', cmap='viridis',
            vmin=abs(phi).min(), vmax=abs(phi).max(), aspect='auto',
            origin='lower', extent=[xs[0], xs[-1], xs[0], xs[-1]])
    cbar = plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$\tilde{\phi}$')

    diffs = abs(phi - solution)

    plt.subplot(122)
    plt.imshow(diffs, interpolation='nearest', cmap='viridis',
            vmin=0, vmax=abs(diffs).max(), aspect='auto',
            origin='lower', extent=[xs[0], xs[-1], xs[0], xs[-1]])
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$|\tilde{\phi} - \phi|$')

    plt.subplots_adjust(bottom=0.12, left=0.06, right=0.96, hspace=0.1)

    plt.savefig('test-problem.png', dpi=120)


if __name__ == '__main__':
    main()

