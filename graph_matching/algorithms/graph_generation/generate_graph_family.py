"""This module contains code to generate graph family
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import graph_matching.algorithms.graph_generation.generate_noisy_graph as generate_noisy_graph
from graph_matching.utils.graph.graph_tools import *
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
    list_ground_truth = []

    min_geo = []
    selected_graphs = []
    selected_ground_truth = []

    graph_index = 0
    while graph_index < nb_sample_graphs:

        ground_truth, noisy_graph = generate_noisy_graph.run(
            reference_graph,
            nb_vertices,
            noise_node,
            noise_edge
        )

        if nx.is_connected(noisy_graph):
            # Add id to edge
            add_integer_id_to_edges(noisy_graph)
            # Save the graph
            list_noisy_graphs.append(noisy_graph)
            # Save all ground-truth for later selecting the selected graphs
            list_ground_truth.append(ground_truth)
            z = mean_edge_len(noisy_graph)
            if min(z) > 7.0:
                selected_graphs.append(noisy_graph)  # select the noisy graph.
                selected_ground_truth.append(ground_truth)  # and its corresponding ground-truth.
                graph_index += 1
                min_geo.append(min(z))


    sorted_zipped_lists = zip(min_geo, selected_graphs, selected_ground_truth)
    sorted_zipped_lists = sorted(sorted_zipped_lists, reverse=True)

    sorted_graphs = []
    sorted_ground_truth = []

    for l, m, n in sorted_zipped_lists:
        sorted_graphs.append(m)
        sorted_ground_truth.append(n)

    ground_truth_perm_to_ref = sorted_ground_truth[:nb_sample_graphs]

    # We generate the ground_truth permutation between graphs

    gtp = ground_truth_labeling(ground_truth_perm_to_ref=ground_truth_perm_to_ref, nb_graphs=nb_sample_graphs)

    return sorted_graphs[:nb_sample_graphs], gtp, ground_truth_perm_to_ref
