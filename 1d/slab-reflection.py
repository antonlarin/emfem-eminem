#!/usr/bin/env python

from __future__ import division
import math, cmath

import numpy as np
import matplotlib.pyplot as plt

from fem_1d import BoundaryCondition, fem_1d, LINEAR, QUADRATIC, CUBIC


def compute_reflection_ez(theta, order, element_count):
    # physical problem setup
    L = 1
    mu_r = lambda x: 2 - .1j
    k0 = 10 * math.pi / L
    eps_r = lambda x: 4 + (2 - .1j) * (1 - x / L)**2
    E0 = 1

    # FEM setup
    l = L / element_count
    mesh = np.linspace(0, L, order * element_count + 1)
    alpha = lambda x: 1 / mu_r(x)
    beta = lambda x: -k0**2 * (eps_r(x) - 1 / mu_r(x) * math.sin(theta)**2)
    f = lambda x: 0

    # boundary condition setup
    left_bc = BoundaryCondition('dirichlet', p=0)
    right_bc = BoundaryCondition(
        'mixed',
        gamma=(k0 * math.cos(theta) * 1j),
        q=(2j * k0 * math.cos(theta) * E0 *
            cmath.exp(1j * k0 * L * math.cos(theta)))
    )

    E_zs = fem_1d(
            order,
            mesh,
            alpha,
            beta,
            f,
            left_bc,
            right_bc,
            dtype='complex')

    return ((E_zs.phis[-1] - E0 * cmath.exp(1j * k0 * L * math.cos(theta))) /
        (E0 * cmath.exp(-1j * k0 * L * math.cos(theta))))


def compute_reflection_hz(theta, order, element_count):
    # physical problem setup
    L = 1
    mu_r = lambda x: 2 - .1j
    k0 = 10 * math.pi / L
    eps_r = lambda x: 4 + (2 - .1j) * (1 - x / L)**2
    H0 = 1

    # FEM setup
    l = L / element_count
    mesh = np.linspace(0, L, order * element_count + 1)
    alpha = lambda x: 1 / eps_r(x)
    beta = lambda x: -k0**2 * (mu_r(x) - 1 / eps_r(x) * math.sin(theta)**2)
    f = lambda x: 0

    # boundary condition setup
    left_bc = BoundaryCondition('neumann', q=0)
    right_bc = BoundaryCondition(
        'mixed',
        gamma=k0 * math.cos(theta) * 1j,
        q=(2j * k0 * math.cos(theta) * H0 *
            cmath.exp(1j * k0 * L * math.cos(theta)))
    )

    H_zs = fem_1d(
            order,
            mesh,
            alpha,
            beta,
            f,
            left_bc,
            right_bc,
            dtype='complex')

    return ((H_zs.phis[-1] - H0 * cmath.exp(1j * k0 * L * math.cos(theta))) /
        (H0 * cmath.exp(-1j * k0 * L * math.cos(theta))))


def main():
    runs = 30
    thetas = np.linspace(0, math.pi / 2, runs)
    rs_linear = np.zeros(runs)
    rs_quadratic = np.zeros(runs)
    rs_cubic = np.zeros(runs)
    for i, theta in enumerate(thetas):
        rs_linear[i] = abs(compute_reflection_ez(theta, LINEAR, 50))**2
        rs_quadratic[i] = abs(compute_reflection_ez(theta, QUADRATIC, 50))**2
        rs_cubic[i] = abs(compute_reflection_ez(theta, CUBIC, 50))**2

    degrees = map(lambda theta: 180 * theta / math.pi, thetas)

    # plot results
    plt.plot(degrees, rs_linear, 'b', label='FEM linear')
    plt.plot(degrees, rs_quadratic, 'r', label='FEM quadratic')
    plt.plot(degrees, rs_cubic, 'g', label='FEM cubic')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(degrees[0], degrees[-1])
    plt.xlabel(r'$\theta$, degrees', fontsize=18)
    plt.ylabel('Reflection coefficient', fontsize=18)
    plt.savefig('slab-reflection-ez-polarization.png', dpi=120)

    plt.clf()
    for i, theta in enumerate(thetas):
        rs_linear[i] = abs(compute_reflection_hz(theta, LINEAR, 50))**2
        rs_quadratic[i] = abs(compute_reflection_hz(theta, QUADRATIC, 50))**2
        rs_cubic[i] = abs(compute_reflection_hz(theta, CUBIC, 50))**2

    # plot results
    plt.plot(degrees, rs_linear, 'b', label='FEM linear')
    plt.plot(degrees, rs_quadratic, 'r', label='FEM quadratic')
    plt.plot(degrees, rs_cubic, 'g', label='FEM cubic')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(degrees[0], degrees[-1])
    plt.xlabel(r'$\theta$, degrees', fontsize=18)
    plt.ylabel('Reflection coefficient', fontsize=18)
    plt.savefig('slab-reflection-hz-polarization.png', dpi=120)


if __name__ == '__main__':
    main()

