"""This module contains function tools for graph generation

..moduleauthor:: Marius Thorre
"""

import sys

sys.path.append("graph_matching/utils/graph")

import trimesh
import random
import networkx as nx
from sphere import *
import graph_processing


def tri_from_hull(vertices: int):
    """ Compute faces from vertices using trimesh convex hul
    :param vertices:
    :return:
    """
    return trimesh.Trimesh(vertices=vertices, process=False).convex_hull


def edge_len_threshold(graph: nx.Graph, thr: nx.Graph()):
    """ Adds a percentage of edges
    :param nx.Graph graph:
    :param int thr:
    :return:
    """
    return random.sample(list(graph.edges), round(len(graph.edges) * thr))


def compute_beta(alpha: float, n: int, mean: float):
    """
    :param alpha:
    :param n:
    :param mean:
    :return:
    """
    return (1 - mean / n) / (mean / n) * alpha


def compute_alpha(n: int, mean: float, variance: float):
    """
    :param n:
    :param mean:
    :param variance:
    :return:
    """
    ratio = (1 - mean / n) / (mean / n)
    alpha = ((1 + ratio) ** 2 * variance - n ** 2 * ratio) / (n * ratio * (1 + ratio) - variance * (1 + ratio) ** 3)
    return alpha


def compute_edge_attribute(
        noisy_graph: nx.Graph,
        radius: float
) -> dict:
    """ Add the edge attributes to the graph
    :param noisy_graph:
    :param radius:
    :return: Edge attributes
    :rtype dict
    """
    edge_attribute = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in noisy_graph.edges:
        # We calculate the geodesic distance
        end_a = noisy_graph.nodes()[edge[0]]["coord"]
        end_b = noisy_graph.nodes()[edge[1]]["coord"]
        geodesic_dist = graph_processing.get_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1
    return edge_attribute


def von_mises_sampling(
        nb_vertices: int,
        original_graph: nx.Graph,
        sigma_noise_nodes: int
) -> list:
    """ Perturbed the coordinates
    :param nb_vertices:
    :param original_graph:
    :param sigma_noise_nodes:
    :return Noisy coord:
    :rtype: list
    """
    noisy_coord = []
    for index in range(nb_vertices):
        # Sampling from Von Mises - Fisher distribution
        original_coord = original_graph.nodes[index]["coord"]
        mean_original = original_coord / np.linalg.norm(original_coord)  # convert to mean unit vector
        noisy_coordinate = Sphere().sample(1, distribution='vMF', mu=mean_original,
                                           kappa=sigma_noise_nodes).sample[0]

        noisy_coordinate = noisy_coordinate * np.linalg.norm(original_coord)  # rescale to original size
        noisy_coord.append(noisy_coordinate)
    return noisy_coord


def get_nearest_neighbors(
        original_coordinates: list,
        list_neighbors: list,
        radius: float,
        nb_to_take: int = 10
) -> list:
    """
    Return the nb_to_take nearest neighbors (in term of geodesic distance) given a set
    of original coordinates and a list of tuples where the first term is the label
    of the node and the second the associated coordinates
    :param original_coordinates:
    :param list_neighbors:
    :param radius:
    :param nb_to_take:
    :return: Sphere geodesic distances
    """
    distances = [(i, graph_processing.get_geodesic_distance_sphere(original_coordinates, current_coordinates, radius))
                 for i, current_coordinates in list_neighbors]
    distances.sort(key=lambda x: x[1])

    return distances[:nb_to_take]


def mean_edge_len(
        graph: nx.Graph
):
    """
    :param graph:
    :return:
    """
    return [z['geodesic_distance'] for x, y, z in list(graph.edges.data())]


def get_in_between_perm_matrix(
        perm_mat_1,
        perm_mat_2
) -> dict:
    """
    Given two permutation from noisy graphs to a reference graph,
    Return the permutation matrix to go from one graph to the other
    :param perm_mat_1:
    :param perm_mat_2:
    :return:
    """
    result_perm = {}
    for i in range(len(perm_mat_1)):
        if perm_mat_1[i] == -1:
            continue
        for j in range(len(perm_mat_2)):
            if perm_mat_2[j] == -1:
                continue

            if perm_mat_1[i] == perm_mat_2[j]:
                result_perm[i] = j

    return result_perm


def extract_ground_truth_permutation(
        noisy_graph: nx.Graph,
        noisy_coord_all: list,
        sphere_random_sampling: list
) -> list:
    """ Extract ground truth permutation
    :param noisy_graph:
    :param noisy_coord_all:
    :param sphere_random_sampling:
    :return:
    """
    ground_truth_permutation = []

    for i in range(len(noisy_graph.nodes)):
        for j in range(len(noisy_coord_all)):  # upto the indexes of outliers
            if np.linalg.norm(noisy_coord_all[j] - noisy_graph.nodes[i]['coord']) == 0.:
                ground_truth_permutation.append(j)
                continue

            elif j == len(noisy_coord_all) - 1.:
                for outlier in sphere_random_sampling:
                    if np.linalg.norm(outlier - noisy_graph.nodes[i]['coord']) == 0.:
                        noisy_graph.nodes[i]['is_outlier'] = True

                        ground_truth_permutation.append(-1)

    return ground_truth_permutation


def add_integer_id_to_edges(
        graph: nx.Graph
):
    """ Given a graph, add an attribute "id" to each edge that is a unique integer id"""

    dict_attributes = {}
    id_counter = 0
    for edge in graph.edges:
        dict_attributes[edge] = {"id": id_counter}
        id_counter += 1
    nx.set_edge_attributes(graph, dict_attributes)


def ground_truth_labeling(
        ground_truth_perm_to_ref: list,
        nb_graphs: int
) -> dict:
    """ Get Ground truth labeling
    :param ground_truth_perm_to_ref:
    :return:
    """
    ground_truth_perm = {}
    for i_graph in range(nb_graphs):

        for j_graph in range(nb_graphs):
            ground_truth_perm[str(i_graph) + ',' + str(j_graph)] = get_in_between_perm_matrix(
                ground_truth_perm_to_ref[i_graph], ground_truth_perm_to_ref[j_graph])
    return ground_truth_perm
