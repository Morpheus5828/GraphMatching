"""This module contains code to generate noisy graph
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""
import random

import networkx as nx
import numpy as np

from graph_matching.utils.graph_tools import *
import \
    graph_matching.algorithms.graph_generation.generate_nb_outliers_and_nb_supress as generate_nb_outliers_and_nb_supress
import graph_matching.algorithms.graph_generation.generate_sphere_random_sampling as generate_sphere_random_sampling
from resources.slam import topology


def run(
        original_graph: nx.Graph,
        nb_vertices: int,
        sigma_noise_nodes: int = 1,
        sigma_noise_edges: int = 1,
        radius: int = 100,
        label_outlier=-1
) -> (list, nx.Graph):
    """ Generate a noisy graph
    :param original_graph:
    :param nb_vertices:
    :param sigma_noise_nodes:
    :param sigma_noise_edges:
    :param radius:
    :return:
    """

    sample_nodes = von_mises_sampling(nb_vertices, original_graph, sigma_noise_edges)

    nb_outliers, nb_supress = generate_nb_outliers_and_nb_supress.run(nb_vertices)

    random_keys = []
    for i in range(nb_supress):
        random_key = random.choice(list(sample_nodes.items()))[0]
        random_keys.append(random_key)
        del sample_nodes[random_key]

    #create nb_outliers
    outliers = generate_sphere_random_sampling.run(vertex_number=nb_outliers, radius=radius)
    for outlier in outliers:
        random_key = random.choice(list(sample_nodes.items()))[0]
        sample_nodes[random_key] = {"coord": outlier, 'is_outlier': True, "label": -1}


    sample_nodes = dict(sorted(sample_nodes.items(), key=lambda item: (item[1]['label'] >= 0, item[1]['label'])))
    print(sample_nodes)
    all_coord = np.array([node["coord"] for node in sample_nodes.values()])
    compute_noisy_edges = tri_from_hull(all_coord)  # take all peturbated coord and comp conv hull.
    adj_matrix = topology.adjacency_matrix(compute_noisy_edges)  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_array(adj_matrix.todense())
    #noisy_graph = nx.path_graph(len(sample_nodes))
    nx.set_node_attributes(noisy_graph, sample_nodes)
    nx.set_edge_attributes(noisy_graph, 1.0, name="weight")

    edge_to_remove = edge_len_threshold(noisy_graph, 0.10)
    noisy_graph.remove_edges_from(edge_to_remove)

    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))



    return noisy_graph
