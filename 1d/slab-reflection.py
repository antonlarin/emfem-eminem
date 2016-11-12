#!/usr/bin/env python

from __future__ import print_function, division
import math, cmath

import numpy as np
import numpy.matlib
import scipy.linalg
import matplotlib.pyplot as plt


def solve_with_fem(
        element_count,
        node_count,
        alphas,
        betas,
        ls,
        fs,
        left_bc,
        right_bc):

    A = np.matlib.zeros((3, node_count), dtype=np.complex)
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

    b = np.zeros(node_count, dtype=np.complex)
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

    return scipy.linalg.solve_banded((1, 1), A, b)


def compute_reflection_ez(theta, element_count):
    # physical problem setup
    L = 1
    mu_r = 2 - .1j
    k0 = 10 * math.pi / L
    eps_r = lambda x: 4 + (2 - .1j) * (1 - x / L)**2
    E0 = 1

    # FEM setup
    l = L / element_count
    node_count = element_count + 1
    ls = [l] * element_count
    element_middles = [l * (0.5 + i) for i in range(element_count)]
    alphas = [1 / mu_r] * element_count
    betas = map(
            lambda x: -k0**2 * (eps_r(x) - 1 / mu_r * math.sin(theta)**2),
            element_middles
    )
    fs = [0] * element_count

    # boundary condition setup
    left_bc = { 'type' : 'dirichlet', 'p' : 0 }
    right_bc = {
        'type' : 'mixed',
        'gamma' : k0 * math.cos(theta) * 1j,
        'q' : (2j * k0 * math.cos(theta) * E0 *
            cmath.exp(1j * k0 * L * math.cos(theta)))
    }

    E_zs = solve_with_fem(
            element_count,
            node_count,
            alphas,
            betas,
            ls,
            fs,
            left_bc,
            right_bc)

    return ((E_zs[-1] - E0 * cmath.exp(1j * k0 * L * math.cos(theta))) /
        (E0 * cmath.exp(-1j * k0 * L * math.cos(theta))))


def compute_reflection_hz(theta, element_count):
    # physical problem setup
    L = 1
    mu_r = 2 - .1j
    k0 = 10 * math.pi / L
    eps_r = lambda x: 4 + (2 - .1j) * (1 - x / L)**2
    H0 = 1

    # FEM setup
    l = L / element_count
    node_count = element_count + 1
    ls = [l] * element_count
    element_middles = [l * (0.5 + i) for i in range(element_count)]
    alphas = map(lambda x: 1 / eps_r(x), element_middles)
    betas = map(
            lambda x: -k0**2 * (mu_r - 1 / eps_r(x) * math.sin(theta)**2),
            element_middles
    )
    fs = [0] * element_count

    # boundary condition setup
    left_bc = { 'type' : 'mixed', 'gamma' : 0, 'q' : 0 }
    right_bc = {
        'type' : 'mixed',
        'gamma' : k0 * math.cos(theta) * 1j,
        'q' : (2j * k0 * math.cos(theta) * H0 *
            cmath.exp(1j * k0 * L * math.cos(theta)))
    }

    H_zs = solve_with_fem(
            element_count,
            node_count,
            alphas,
            betas,
            ls,
            fs,
            left_bc,
            right_bc)

    return ((H_zs[-1] - H0 * cmath.exp(1j * k0 * L * math.cos(theta))) /
        (H0 * cmath.exp(-1j * k0 * L * math.cos(theta))))


def main():
    runs = 50
    thetas = np.linspace(0, math.pi / 2, runs)
    rs_50 = np.zeros(runs)
    rs_100 = np.zeros(runs)
    for i, theta in enumerate(thetas):
        rs_50[i] = abs(compute_reflection_ez(theta, 50))**2
        rs_100[i] = abs(compute_reflection_ez(theta, 100))**2

    degrees = map(lambda theta: 180 * theta / math.pi, thetas)

    # plot results
    plt.plot(degrees, rs_50, 'b', label='FEM 50 elems')
    plt.plot(degrees, rs_100, 'r', label='FEM 100 elems')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(degrees[0], degrees[-1])
    plt.xlabel(r'$\theta$, degrees', fontsize=18)
    plt.ylabel('Reflection coefficient', fontsize=18)
    plt.savefig('slab-reflection-ez-polarization.png', dpi=120)

    plt.clf()
    for i, theta in enumerate(thetas):
        rs_50[i] = abs(compute_reflection_hz(theta, 50))**2
        rs_100[i] = abs(compute_reflection_hz(theta, 100))**2

    # plot results
    plt.plot(degrees, rs_50, 'b', label='FEM 50 elems')
    plt.plot(degrees, rs_100, 'r', label='FEM 100 elems')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(degrees[0], degrees[-1])
    plt.xlabel(r'$\theta$, degrees', fontsize=18)
    plt.ylabel('Reflection coefficient', fontsize=18)
    plt.savefig('slab-reflection-hz-polarization.png', dpi=120)


if __name__ == '__main__':
    main()

