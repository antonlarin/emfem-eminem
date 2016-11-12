#!/usr/bin/env python

from __future__ import print_function, division
import math

import numpy as np
import numpy.matlib
import scipy.linalg
import matplotlib.pyplot as plt

# equation setup
L = 1
element_count = 20
node_count = element_count + 1
alphas = [0.7] * element_count
betas = [0.3] * element_count
l = L / element_count
ls = [l] * element_count
def f(x):
    s = math.sin(2 * math.pi * x)
    c = math.cos(2 * math.pi * x)
    return math.exp(x) * (0.3 * c +
            0.7 * (-c + 4 * math.pi * (s + math.pi * c)))

fs = [f((0.5 + i) * l) for i in range(element_count)]

# boundary condition setup
left_bc = { 'type' : 'dirichlet', 'p' : 1 }
right_bc = { 'type' : 'mixed', 'gamma' : -2, 'q' : -1.3 * math.e }


def main():
    A = np.matlib.zeros((3, node_count))
    for i in range(element_count):
        local_k_11 = alphas[i] / ls[i] + betas[i] * ls[i] / 3
        local_k_22 = local_k_11
        A[1, i] += local_k_11
        A[1, i + 1] += local_k_22

    for i in range(element_count):
        local_k_12 = -alphas[i] / ls[i] + betas[i] * ls[i] / 6
        local_k_21 = local_k_12
        A[0, i + 1] = local_k_12
        A[2, i] = local_k_21

    b = np.zeros(node_count)
    for i in range(element_count):
        local_b_1 = fs[i] * ls[i] / 2
        local_b_2 = local_b_1
        b[i] += local_b_1
        b[i + 1] += local_b_2

    # left bc
    if left_bc['type'] == 'dirichlet':
        A[1, 0] = 1
        b[0] = left_bc['p']
        b[1] -= A[2, 0] * left_bc['p']
        A[0, 1] = 0
        A[2, 0] = 0
    elif left_bc['type'] == 'mixed':
        A[1, 0] += left_bc['gamma']
        b[0] += left_bc['q']
    else:
        raise ValueError('Unsupported BC type')

    # right bc
    if right_bc['type'] == 'dirichlet':
        A[1, -1] = 1
        b[-1] = right_bc['p']
        b[-2] -= A[0, -1] * right_bc['p']
        A[0, -1] = 0
        A[2, -2] = 0
    elif right_bc['type'] == 'mixed':
        A[1, -1] += right_bc['gamma']
        b[-1] += right_bc['q']
    else:
        raise ValueError('Unsupported BC type')

    phis = scipy.linalg.solve_banded((1, 1), A, b)

    # plot results
    xs = [ sum(ls[:i]) for i in range(node_count) ]
    plt.plot(xs, phis, 'b', label='Numerical')

    solution_xs = np.linspace(0, L, 100)
    solution = map(lambda x: math.exp(x) * math.cos(2 * math.pi * x), solution_xs)
    plt.plot(solution_xs, solution, 'r', label='Analytical')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$\phi$', fontsize=18)
    plt.savefig('exp(x)*cos(2pi*x).png', dpi=120)


if __name__ == '__main__':
    main()

