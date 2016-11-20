from __future__ import division
import math

import numpy as np
import scipy.linalg


class Problem2D(object):
    def __init__(self, alpha_x, alpha_y, beta, f, gamma, q, p):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta = beta
        self.f = f
        self.gamma = gamma
        self.q = q
        self.p = p


def _map_to_reference_element(triangle, x, y):
    v1 = triangle.v1
    v2 = triangle.v2
    v3 = triangle.v3
    area = triangle.area
    xi = 0.5 * ((v2.y - v3.y) * x + (v3.x - v2.x) * y +
            v2.x * v3.y - v3.x * v2.y) / area
    eta = 0.5 * ((v3.y - v1.y) * x + (v1.x - v3.x) * y +
            v3.x * v1.y - v1.x * v3.y) / area
    return xi, eta

class FemSolution(object):
    def __init__(self, triangles, phis):
        self.triangles = triangles
        self.phis = phis

    def __call__(self, x, y):
        for element in self.triangles:
            if element.contains_point(x, y):
                xi, eta = _map_to_reference_element(element, x, y)

                return (xi * self.phis[element.v1.idx] +
                        eta * self.phis[element.v2.idx] +
                        (1 - xi - eta) * self.phis[element.v3.idx])

        return self.phis.min()


def assemble_system(problem, mesh):
    node_count = len(mesh.nodes)
    K = np.zeros((node_count, node_count))
    b = np.zeros(node_count)

    for element in mesh.elements:
        K_e, b_e = compute_system_on_element(problem, element)

        for i, node1 in enumerate(element.nodes):
            for j, node2 in enumerate(element.nodes):
                K[node1.idx, node2.idx] += K_e[i, j]

            b[node1.idx] += b_e[i]

    K, b = impose_third_kind_conditions(problem, mesh, K, b)
    K, b = impose_continuity_conditions(mesh, K, b)
    K, b = impose_dirichlet_conditions(problem, mesh, K, b)

    return K, b

def compute_system_on_element(problem, element):
    K_e = np.zeros((3, 3))
    b_e = np.zeros(3)
    delta_e = element.area

    alpha_x_e = element.average(problem.alpha_x)
    alpha_y_e = element.average(problem.alpha_y)
    beta_e = element.average(problem.beta)
    f_e = element.average(problem.f)

    basis_derivatives_by_x, basis_derivatives_by_y = \
            element.basis_derivatives()

    for i in range(3):
        for j in range(3):
            K_e[i, j] = (
                    0.25 * (alpha_x_e * basis_derivatives_by_x[i] * 
                                basis_derivatives_by_x[j] +
                            alpha_y_e * basis_derivatives_by_y[i] *
                                basis_derivatives_by_y[j]) / delta_e +
                    delta_e * beta_e / 12
            )

        K_e[i, i] += delta_e * beta_e / 12
        b_e[i] = delta_e * f_e / 3

    return K_e, b_e

def impose_third_kind_conditions(problem, mesh, K, b):
    for edge in mesh.third_kind_edges:
        gamma_s = edge.average(problem.gamma)
        q_s = edge.average(problem.q)
        l_s = edge.length

        for node1 in edge.nodes:
            for node2 in edge.nodes:
                K[node1.idx, node2.idx] += gamma_s * l_s / 6
            K[node1.idx, node1.idx] += gamma_s * l_s / 6
            b[node1.idx] += q_s * l_s / 2

    return K, b


def impose_continuity_conditions(mesh, K, b):
    for node_idx1, node_idx2 in mesh.discontinuity_node_index_pairs:
        for i in range(K.shape[0]):
            K[node_idx2, i] = 0

        K[node_idx2, node_idx1] = 1
        K[node_idx2, node_idx2] = -1
        b[node_idx2] = 0

    return K, b


def impose_dirichlet_conditions(problem, mesh, K, b):
    for node in mesh.dirichlet_nodes:
        p_value = problem.p(node.x, node.y)

        for i in range(K.shape[1]):
            K[node.idx, i] = 0

        for i in range(b.shape[0]):
            b[i] -= K[i, node.idx] * p_value
            K[i, node.idx] = 0

        K[node.idx, node.idx] = 1
        b[node.idx] = p_value

    return K, b


def fem_2d(problem, mesh):
    K, b = assemble_system(problem, mesh)
    
    phis = scipy.linalg.solve(K, b)
    return FemSolution(mesh.elements, phis)

