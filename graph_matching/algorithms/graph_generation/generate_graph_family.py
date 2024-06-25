"""This module contains code to generate graph family
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import graph_matching.algorithms.graph_generation.generate_noisy_graph as generate_noisy_graph
from graph_matching.utils.graph_tools import *
import networkx as nx


def run(
        nb_sample_graphs: int,
        nb_vertices: int,
        ref_graph,
        noise_node=1,
        noise_edge=1,
):
    """
    Generate n noisy graphs from a reference graph alongside the
    :param nb_sample_graphs:
    :param nb_vertices:
    :param ref_graph:
    :param noise_node:
    :param noise_edge:
    :return:
    """
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []

    graph_index = 0
    while graph_index < nb_sample_graphs:

        noisy_graph = generate_noisy_graph.run(
            reference_graph,
            nb_vertices,
            noise_node,
            noise_edge
        )

        if nx.is_connected(noisy_graph):
            list_noisy_graphs.append(noisy_graph)
            graph_index += 1


    return list_noisy_graphs