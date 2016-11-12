from __future__ import division

import numpy as np
import numpy.matlib
import scipy.linalg
import scipy.integrate


class BoundaryCondition(object):
    def __init__(self, bctype, **kwargs):
        self.type = bctype
        if self.type == 'dirichlet':
            self.p = kwargs.get('p', 0.)
        elif self.type == 'neumann':
            self.q = kwargs.get('q', 0.)
        elif self.type == 'mixed':
            self.q = kwargs.get('q', 0.)
            self.gamma = kwargs.get('gamma', 0.)
        else:
            raise ValueError('Unsupported BC type')


def lagrange_1_polynomials(x0, x1):
    return [
            lambda x : (x - x1) / (x0 - x1),
            lambda x : (x - x0) / (x1 - x0)
    ]


def lagrange_1_polynomials_derivatives(x0, x1):
    return [
            lambda x: 1 / (x0 - x1),
            lambda x: 1 / (x1 - x0)
    ]


def complex_quadrature(f, a, b, **kwargs):
    real_integral, _ = scipy.integrate.fixed_quad(lambda x: f(x).real,
            a, b, **kwargs)
    imag_integral, _ = scipy.integrate.fixed_quad(lambda x: f(x).imag,
            a, b, **kwargs)

    return real_integral + imag_integral * 1j;


def fem_1d(
        mesh,
        alpha,
        beta,
        f,
        left_bc,
        right_bc,
        order,
        **kwargs):

    if order == 1:
        return fem_1d_1st_order(mesh, alpha, beta, f, left_bc, right_bc,
                **kwargs)
    # elif order == 2:
        # return fem_1d_2nd_order(mesh, alpha, beta, f, left_bc, right_bc)
    # elif order == 3:
        # return fem_1d_3rd_order(mesh, alpha, beta, f, left_bc, right_bc)
    else:
        raise NotImplementedError


def fem_1d_1st_order(
        mesh,
        alpha,
        beta,
        f,
        left_bc,
        right_bc,
        **kwargs):

    if 'dtype' in kwargs:
        if kwargs['dtype'] == 'real':
            dtype = np.float
            quadrature = scipy.integrate.fixed_quad
        elif kwargs['dtype'] == 'complex':
            dtype = np.complex
            quadrature = complex_quadrature
        else:
            raise ValueError('Can only compute in reals or complex')
    else:
        dtype = np.float

    node_count = len(mesh)
    element_count = node_count - 1

    K = np.matlib.zeros((3, node_count), dtype=dtype)
    b = np.zeros(node_count, dtype=dtype)

    for element in range(element_count):
        x0 = mesh[element]
        x1 = mesh[element + 1]

        K_e = np.matlib.zeros((2,2), dtype=dtype)
        b_e = np.zeros(2, dtype=dtype)
        basis = lagrange_1_polynomials(x0, x1)
        basis_derivatives = lagrange_1_polynomials_derivatives(x0, x1)
        for i in range(2):
            for j in range(i, 2):
                K_e[i, j] = quadrature(
                        lambda x: (alpha(x) * basis_derivatives[i](x) *
                            basis_derivatives[j](x) +
                            beta(x) * basis[i](x) * basis[j](x)),
                        x0, x1, n=3)
                K_e[j, i] = K_e[i, j]
            b_e[i] = quadrature(
                    lambda x: f(x) * basis[i](x),
                    x0, x1, n=3)

        K[1, element] += K_e[0, 0]
        K[0, element + 1] += K_e[0, 1]
        K[1, element + 1] += K_e[1, 1]
        K[2, element] += K_e[1, 0]

        b[element] += b_e[0]
        b[element + 1] += b_e[1]


    # left bc
    if left_bc.type == 'dirichlet':
        K[1, 0] = 1
        b[0] = left_bc.p
        b[1] -= K[2, 0] * left_bc.p
        K[0, 1] = 0
        K[2, 0] = 0
    elif left_bc.type == 'neumann':
        b[0] += left_bc.q
    else: # mixed
        K[1, 0] += left_bc.gamma
        b[0] += left_bc.q

    # right bc
    if right_bc.type == 'dirichlet':
        K[1, -1] = 1
        b[-1] = right_bc.p
        b[-2] -= K[0, -1] * right_bc.p
        K[0, -1] = 0
        K[2, -2] = 0
    elif right_bc.type == 'neumann':
        pass
    else: # mixed
        K[1, -1] += right_bc.gamma
        b[-1] += right_bc.q

    return scipy.linalg.solve_banded((1, 1), K, b)

