#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import matplotlib.pyplot as plt

from fem_1d import fem_1d, BoundaryCondition, LINEAR, QUADRATIC, CUBIC

# equation setup
L = 1
element_count = 60
l = L / element_count
mesh = np.linspace(0, L, element_count + 1)
alpha = lambda x: 0.7j
beta = lambda x: 0.3j
def f(x):
    s = math.sin(2 * math.pi * x)
    c = math.cos(2 * math.pi * x)
    return math.exp(x) * (0.3j * c +
            0.7j * (-c + 4 * math.pi * (s + math.pi * c)))

# boundary condition setup
left_bc = BoundaryCondition('dirichlet', p=1)
right_bc = BoundaryCondition('mixed', gamma=-2, q=math.e * (-2 + 0.7j))


def main():
    phis_1 = fem_1d(LINEAR, mesh, alpha, beta, f, left_bc, right_bc, dtype='complex')
    phis_2 = fem_1d(QUADRATIC, mesh, alpha, beta, f, left_bc, right_bc, dtype='complex')
    phis_3 = fem_1d(CUBIC, mesh, alpha, beta, f, left_bc, right_bc, dtype='complex')

    # plot results
    xs = np.linspace(0, L, 150)
    phis_1_grid = map(lambda x: phis_1(x).real, xs)
    phis_2_grid = map(lambda x: phis_2(x).real, xs)
    phis_3_grid = map(lambda x: phis_3(x).real, xs)

    phis_1_grid_imag = map(lambda x: phis_1(x).imag, xs)
    phis_2_grid_imag = map(lambda x: phis_2(x).imag, xs)
    phis_3_grid_imag = map(lambda x: phis_3(x).imag, xs)

    solution = map(lambda x: math.exp(x) * math.cos(2 * math.pi * x), xs)

    plt.subplot(211)
    plt.plot(xs, phis_1_grid, 'b', label='Numerical linear real')
    plt.plot(xs, phis_2_grid, 'g', label='Numerical quadratic real')
    plt.plot(xs, phis_3_grid, 'm', label='Numerical cubic real')
    plt.plot(xs, phis_1_grid_imag, 'k', label='Numerical linear imaginary')
    plt.plot(xs, phis_2_grid_imag, 'c', label='Numerical quadratic imaginary')
    plt.plot(xs, phis_3_grid_imag, 'y', label='Numerical cubic imaginary')
    plt.plot(xs, solution, 'r', label='Analytical')

    plt.grid()
    plt.legend(loc='best', fontsize=8)
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

