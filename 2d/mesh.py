from __future__ import division
import math


class Mesh(object):
    def __init__(self, node_descriptions, element_descriptions,
            physical_entities):
        self.nodes = _wrap_nodes(node_descriptions)
        self.elements = _wrap_elements(element_descriptions, self.nodes)
        self.dirichlet_nodes = _find_dirichlet_nodes(element_descriptions,
                self.nodes, physical_entities)
        self.third_kind_edges = _find_third_kind_edges(element_descriptions,
                self.nodes, physical_entities)

def _wrap_nodes(node_descriptions):
    wrap_node = lambda idx, x, y, z: Node(x, y, idx - 1)
    wrapped_nodes = [wrap_node(*node_description)
            for node_description in node_descriptions]
    wrapped_nodes.sort(key=lambda node: node.idx)
    return wrapped_nodes


def _wrap_elements(element_descriptions, wrapped_nodes):
    def is_triangle(element_description):
        MSH_3_POINT_TRIANGLE = 2
        element_type = element_description[1]
        return element_type == MSH_3_POINT_TRIANGLE

    def wrap_triangle(
            idx,
            element_type,
            tag_count,
            tags,
            nodes):
        return Triangle(
                wrapped_nodes[nodes[0] - 1],
                wrapped_nodes[nodes[1] - 1],
                wrapped_nodes[nodes[2] - 1])

    triangle_descriptions = filter(is_triangle, element_descriptions)
    return [wrap_triangle(*triangle_description) for
            triangle_description in triangle_descriptions]


def _find_dirichlet_nodes(element_descriptions, wrapped_nodes,
        physical_entities):
    try:
        dirichlet_index = physical_entities['dirichlet-boundary'][1]
    except KeyError:
        return []

    dirichlet_nodes = set()
    MSH_2_POINT_LINE = 1
    for element_description in element_descriptions:
        element_type = element_description[1]
        parent_physical_entity = element_description[3][0]
        node_indices = element_description[4]
        
        if (element_type == MSH_2_POINT_LINE and
                parent_physical_entity == dirichlet_index):
            for idx in node_indices:
                dirichlet_nodes.add(wrapped_nodes[idx - 1])

    return list(dirichlet_nodes)


def _find_third_kind_edges(element_descriptions, wrapped_nodes, physical_entities):
    try:
        third_kind_index = physical_entities['third-kind-boundary'][1]
    except KeyError:
        return []

    third_kind_edges = []
    MSH_2_POINT_LINE = 1
    for element_description in element_descriptions:
        element_type = element_description[1]
        parent_physical_entity = element_description[3][0]
        node_indices = element_description[4]

        if (element_type == MSH_2_POINT_LINE and
                parent_physical_entity == third_kind_index):
            third_kind_edges.append(Edge(
                    wrapped_nodes[node_indices[0] - 1],
                    wrapped_nodes[node_indices[1] - 1]))

    return third_kind_edges


def _triangle_area(v1, v2, v3):
    x1 = v2.x - v1.x
    y1 = v2.y - v1.y

    x2 = v3.x - v1.x
    y2 = v3.y - v1.y

    return 0.5 * abs(x1 * y2 - x2 * y1)

class Triangle(object):
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.nodes = [v1, v2, v3]
        self.area = _triangle_area(v1, v2, v3)

    def center(self):
        center_x = (self.v1.x + self.v2.x + self.v3.x) / 3
        center_y = (self.v1.y + self.v2.y + self.v3.y) / 3
        return center_x, center_y

    def average(self, f):
        return f(*self.center())

    def contains_point(self, x, y):
        left_of_v1_v2 = (
            (self.v2.x - self.v1.x) * (y - self.v1.y) -
            (x - self.v1.x) * (self.v2.y - self.v1.y)) >= 0
        left_of_v2_v3 = (
            (self.v3.x - self.v2.x) * (y - self.v2.y) -
            (x - self.v2.x) * (self.v3.y - self.v2.y)) >= 0
        left_of_v3_v1 = (
            (self.v1.x - self.v3.x) * (y - self.v3.y) -
            (x - self.v3.x) * (self.v1.y - self.v3.y)) >= 0

        return left_of_v1_v2 and left_of_v2_v3 and left_of_v3_v1

    def basis_derivatives(self):
        d_by_dx = [
                self.v2.y - self.v3.y,
                self.v3.y - self.v1.y,
                self.v1.y - self.v2.y
        ]
        d_by_dy = [
                self.v3.x - self.v2.x,
                self.v1.x - self.v3.x,
                self.v2.x - self.v1.x
        ]

        return d_by_dx, d_by_dy


class Edge(object):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.length = math.sqrt((v2.x - v1.x)**2 + (v2.y - v1.y)**2)
        self.nodes = [v1, v2]

    def average(self, f):
        return f(*self.center())

    def center(self):
        center_x = (self.v1.x + self.v2.x) / 2
        center_y = (self.v1.y + self.v2.y) / 2
        return center_x, center_y


class Node(object):
    def __init__(self, x, y, idx=None):
        self.idx = idx
        self.x = x
        self.y = y

def drop_double_quotes(string):
    return string.replace('"', '')

def skip_until_end_of_section(mesh_file, end_tag):
    line = None
    while line != end_tag:
        line = mesh_file.readline().rstrip()

def parse_header(mesh_file):
    skip_until_end_of_section(mesh_file, '$EndMeshFormat')

def parse_physical_entities(mesh_file):
    physical_entities_count = int(mesh_file.readline().rstrip())
    physical_entities = {}
    for _ in range(physical_entities_count):
        dimension, index, name = mesh_file.readline().rstrip().split(' ', 2)
        physical_entities[drop_double_quotes(name)] = (int(dimension),
                int(index))

    skip_until_end_of_section(mesh_file, '$EndPhysicalNames')
    return physical_entities

def parse_nodes(mesh_file):
    nodes_count = int(mesh_file.readline().rstrip())
    nodes = []
    for _ in range(nodes_count):
        index, x, y, z = mesh_file.readline().rstrip().split()
        nodes.append((int(index), float(x), float(y), float(z)))
        
    skip_until_end_of_section(mesh_file, '$EndNodes')
    return nodes

def parse_elements(mesh_file):
    elements_count = int(mesh_file.readline().rstrip())
    elements = []
    for _ in range(elements_count):
        index, element_type, tag_count, rest = \
                mesh_file.readline().rstrip().split(' ', 3)
        tags = []
        for tag_idx in range(int(tag_count)):
            tag, rest = rest.split(' ', 1)
            tags.append(int(tag))

        nodes = map(int, rest.split())

        elements.append((int(index), int(element_type), int(tag_count),
                tags, nodes))

    skip_until_end_of_section(mesh_file, '$EndElements')
    return elements

def load_msh(filename):
    with open(filename, 'rt') as mesh_file:
        line = None
        while line != '':
            line = mesh_file.readline().rstrip()
            if line == '$MeshFormat':
                parse_header(mesh_file)
            elif line == '$PhysicalNames':
                physical_entities = parse_physical_entities(mesh_file)
            elif line == '$Nodes':
                nodes = parse_nodes(mesh_file)
            elif line == '$Elements':
                elements = parse_elements(mesh_file)

    return Mesh(nodes, elements, physical_entities)

