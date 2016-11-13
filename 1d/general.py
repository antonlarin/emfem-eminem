#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import matplotlib.pyplot as plt

from fem_1d import fem_1d, BoundaryCondition, LINEAR, QUADRATIC, CUBIC

# equation setup
L = 1
element_count = 6
l = L / element_count
mesh = np.linspace(0, L, element_count + 1)
alpha = lambda x: 0.7
beta = lambda x: 0.3
def f(x):
    s = math.sin(2 * math.pi * x)
    c = math.cos(2 * math.pi * x)
    return math.exp(x) * (0.3 * c +
            0.7 * (-c + 4 * math.pi * (s + math.pi * c)))

# boundary condition setup
left_bc = BoundaryCondition('dirichlet', p=1)
right_bc = BoundaryCondition('mixed', gamma=-2, q=-1.3 * math.e)


def main():
    phis_1 = fem_1d(LINEAR, mesh, alpha, beta, f, left_bc, right_bc)
    phis_2 = fem_1d(QUADRATIC, mesh, alpha, beta, f, left_bc, right_bc)
    phis_3 = fem_1d(CUBIC, mesh, alpha, beta, f, left_bc, right_bc)

    # plot results
    xs = np.linspace(0, L, 150)
    phis_1_grid = map(phis_1, xs)
    phis_2_grid = map(phis_2, xs)
    phis_3_grid = map(phis_3, xs)

    solution = map(lambda x: math.exp(x) * math.cos(2 * math.pi * x), xs)

    plt.subplot(211)
    plt.plot(xs, phis_1_grid, 'b', label='Numerical linear')
    plt.plot(xs, phis_2_grid, 'g', label='Numerical quadratic')
    plt.plot(xs, phis_3_grid, 'm', label='Numerical cubic')
    plt.plot(xs, solution, 'r', label='Analytical')

    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.xlim(mesh[0], mesh[-1])
    plt.ylabel(r'$\phi$', fontsize=18)

    xi_1_grid = np.abs(np.asarray(phis_1_grid) - np.asarray(solution))
    xi_2_grid = np.abs(np.asarray(phis_2_grid) - np.asarray(solution))
    xi_3_grid = np.abs(np.asarray(phis_3_grid) - np.asarray(solution))

    plt.subplot(212)
    plt.plot(xs, xi_1_grid, 'b', label='Numerical linear')
    plt.plot(xs, xi_2_grid, 'g', label='Numerical quadratic')
    plt.plot(xs, xi_3_grid, 'm', label='Numerical cubic')

    plt.grid()
    plt.yscale('log')
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$\xi$', fontsize=18)

    plt.savefig('test-problem.png', dpi=120)


if __name__ == '__main__':
    main()

