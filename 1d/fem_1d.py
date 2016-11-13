from __future__ import division

import numpy as np
import numpy.matlib
import scipy.linalg
import scipy.integrate

import lagrange_polynomials

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


class FemSolution(object):
    def __init__(self, order, mesh, phis):
        self.order = order
        self.mesh = mesh
        self.element_count = (len(mesh) - 1) // order
        self.phis = phis

    def __call__(self, x):
        for element in range(self.element_count):
            x_es = self.mesh[element * self.order :
                    (element + 1) * self.order + 1]
            if (x_es[0] <= x and x <= x_es[-1]):
                basis = lagrange_polynomials.basis(self.order, *x_es)
                phi_es = self.phis[element * self.order :
                        (element + 1) * self.order + 1]
                phi_of_x = 0
                for N, phi in zip(basis, phi_es):
                    phi_of_x += N(x) * phi

                return phi_of_x

        raise ValueError('x is outside of domain')


def complex_quadrature(f, a, b, **kwargs):
    real_integral, _ = scipy.integrate.fixed_quad(lambda x: f(x).real,
            a, b, **kwargs)
    imag_integral, _ = scipy.integrate.fixed_quad(lambda x: f(x).imag,
            a, b, **kwargs)

    return real_integral + imag_integral * 1j;


LINEAR = 1
QUADRATIC = 2
CUBIC = 3


def fem_1d(
        order,
        mesh,
        alpha,
        beta,
        f,
        left_bc,
        right_bc,
        **kwargs):
    
    if not order in ( LINEAR, QUADRATIC, CUBIC ):
        raise NotImplementedError('Unsupported element order')

    if 'dtype' in kwargs:
        if kwargs['dtype'] == 'real':
            dtype = np.float
        elif kwargs['dtype'] == 'complex':
            dtype = np.complex
        else:
            raise ValueError('Can only compute in reals or complex')
    else:
        dtype = np.float

    if dtype is np.complex:
        quadrature = complex_quadrature
    else:
        quadrature = (lambda f, a, b, **kwargs:
                scipy.integrate.fixed_quad(f, a, b, **kwargs)[0])

    node_count = len(mesh)
    element_count = (node_count - 1) // order
    element_node_count = order + 1

    vec_f = np.vectorize(f)

    K = np.matlib.zeros((2 * order + 1, node_count), dtype=dtype)
    b = np.zeros(node_count, dtype=dtype)

    for element in range(element_count):
        element_shift = element * order
        element_xs = mesh[element_shift :
                element_shift + order + 1]

        K_e = np.matlib.zeros((element_node_count, element_node_count),
                dtype=dtype)
        b_e = np.zeros(element_node_count, dtype=dtype)
        basis = lagrange_polynomials.basis(order, *element_xs)
        basis_derivatives = lagrange_polynomials.basis_derivatives(
                order, *element_xs)
        for i in range(element_node_count):
            for j in range(i, element_node_count):
                K_e[i, j] = quadrature(
                        lambda x: (
                            alpha(x) * basis_derivatives[i](x) *
                            basis_derivatives[j](x) +
                            beta(x) * basis[i](x) * basis[j](x)
                            ),
                        element_xs[0], element_xs[-1])
                K_e[j, i] = K_e[i, j]
            b_e[i] = quadrature(
                lambda x: vec_f(x) * basis[i](x),
                element_xs[0], element_xs[-1]
            )

        for i in range(element_node_count):
            for j in range(element_node_count):
                diagonal = i - j
                K[diagonal + order, j + element_shift] += K_e[i, j]

            b[element_shift + i] += b_e[i]

    # left bc
    if left_bc.type == 'dirichlet':
        K[order, 0] = 1
        b[0] = left_bc.p
        for i in range(1, order + 1):
            b[i] -= K[order + i, 0] * left_bc.p
            K[order + i, 0] = 0
            K[order - i, i] = 0
    elif left_bc.type == 'neumann':
        b[0] += left_bc.q
    else: # mixed
        K[order, 0] += left_bc.gamma
        b[0] += left_bc.q

    # right bc
    if right_bc.type == 'dirichlet':
        K[order, -1] = 1
        b[-1] = right_bc.p
        for i in range(1, order + 1):
            b[-1 - i] -= K[order - i, -1] * right_bc.p
            K[order - i, -1] = 0
            K[order + i, -1 - i] = 0
    elif right_bc.type == 'neumann':
        b[-1] += right_bc.q
    else: # mixed
        K[order, -1] += right_bc.gamma
        b[-1] += right_bc.q

    phis = scipy.linalg.solve_banded((order, order), K, b)
    return FemSolution(order, mesh, phis)

