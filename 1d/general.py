#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import matplotlib.pyplot as plt

from fem_1d import fem_1d, BoundaryCondition, LINEAR, QUADRATIC

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
    phis = fem_1d(mesh, alpha, beta, f, left_bc, right_bc, LINEAR)
    phis_2 = fem_1d(mesh, alpha, beta, f, left_bc, right_bc, QUADRATIC)

    # plot results
    plt.plot(mesh, phis, 'b', label='Numerical 1 order')
    plt.plot(mesh, phis_2, 'g', label='Numerical 2 order')

    solution_xs = np.linspace(0, L, 100)
    solution = map(lambda x: math.exp(x) * math.cos(2 * math.pi * x),
            solution_xs)
    plt.plot(solution_xs, solution, 'r', label='Analytical')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(mesh[0], mesh[-1])
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$\phi$', fontsize=18)
    plt.savefig('exp(x)*cos(2pi*x).png', dpi=120)


if __name__ == '__main__':
    main()

