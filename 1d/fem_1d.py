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
    elif order == 2:
        return fem_1d_2nd_order(mesh, alpha, beta, f, left_bc, right_bc)
    # elif order == 3:
        # return fem_1d_3rd_order(mesh, alpha, beta, f, left_bc, right_bc)
    else:
        raise NotImplementedError

LINEAR = 1
QUADRATIC = 2
CUBIC = 3


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
    element_count = node_count - 1

    vec_f = np.vectorize(f)

    K = np.matlib.zeros((3, node_count), dtype=dtype)
    b = np.zeros(node_count, dtype=dtype)

    for element in range(element_count):
        x0 = mesh[element]
        x1 = mesh[element + 1]

        K_e = np.matlib.zeros((2,2), dtype=dtype)
        b_e = np.zeros(2, dtype=dtype)
        basis = lagrange_polynomials.basis(1, x0, x1)
        basis_derivatives = lagrange_polynomials.basis_derivatives(1, x0, x1)
        for i in range(2):
            for j in range(i, 2):
                K_e[i, j] = quadrature(
                        lambda x: (alpha(x) * basis_derivatives[i](x) *
                            basis_derivatives[j](x) +
                            beta(x) * basis[i](x) * basis[j](x)),
                        x0, x1, n=3)
                K_e[j, i] = K_e[i, j]
            b_e[i] = quadrature(
                    lambda x: vec_f(x) * basis[i](x),
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

    phis = scipy.linalg.solve_banded((1, 1), K, b)
    return FemSolution(1, mesh, phis)

def fem_1d_2nd_order(
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
    element_count = (node_count - 1) // 2

    vec_f = np.vectorize(f)

    K = np.matlib.zeros((5, node_count), dtype=dtype)
    b = np.zeros(node_count, dtype=dtype)

    for element in range(element_count):
        element_shift = element * 2
        x0 = mesh[element_shift]
        x1 = mesh[element_shift + 1]
        x2 = mesh[element_shift + 2]

        K_e = np.matlib.zeros((3,3), dtype=dtype)
        b_e = np.zeros(3, dtype=dtype)
        basis = lagrange_polynomials.basis(2, x0, x1, x2)
        basis_derivatives = \
                lagrange_polynomials.basis_derivatives(2, x0, x1, x2)
        for i in range(3):
            for j in range(i, 3):
                K_e[i, j] = quadrature(
                        lambda x: (alpha(x) * basis_derivatives[i](x) *
                            basis_derivatives[j](x) +
                            beta(x) * basis[i](x) * basis[j](x)),
                        x0, x2, n=3)
                K_e[j, i] = K_e[i, j]
            b_e[i] = quadrature(
                    lambda x: vec_f(x) * basis[i](x),
                    x0, x2, n=3)

        for i in range(3):
            for j in range(3):
                diagonal = i - j
                K[diagonal + 2, j + element_shift] += K_e[i, j]

            b[i + element_shift] += b_e[i]

    # left bc
    if left_bc.type == 'dirichlet':
        K[2, 0] = 1
        b[0] = left_bc.p
        b[1] -= K[3, 0] * left_bc.p
        b[2] -= K[4, 0] * left_bc.p
        K[0, 2] = 0
        K[1, 1] = 0
        K[3, 0] = 0
        K[4, 0] = 0
    elif left_bc.type == 'neumann':
        b[0] += left_bc.q
    else: # mixed
        K[2, 0] += left_bc.gamma
        b[0] += left_bc.q

    # right bc
    if right_bc.type == 'dirichlet':
        K[2, -1] = 1
        b[-1] = right_bc.p
        b[-2] -= K[1, -1] * right_bc.p
        b[-3] -= K[0, -1] * right_bc.p
        K[3, -2] = 0
        K[4, -3] = 0
        K[1, -1] = 0
        K[0, -1] = 0
    elif right_bc.type == 'neumann':
        pass
    else: # mixed
        K[2, -1] += right_bc.gamma
        b[-1] += right_bc.q

    phis = scipy.linalg.solve_banded((2, 2), K, b)
    return FemSolution(2, mesh, phis)

