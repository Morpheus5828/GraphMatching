"""This module contains code to generate graph family
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""


from tqdm.auto import tqdm
import graph_matching.graph_generation.generate_noisy_graph as generate_noisy_graph
from graph_matching.utils.graph.graph_tools import *
import networkx as nx


def run(
        nb_sample_graphs: int,
        nb_graphs: int,
        nb_vertices: int,
        radius: float,
        nb_outliers: int,
        ref_graph,
        noise_node=1,
        noise_edge=1,
        nb_neighbors_to_consider: int = 10
):
    """
    Generate n noisy graphs from a reference graph alongside the
	ground truth permutation matrices.
    :param nb_sample_graphs:
    :param nb_graphs:
    :param nb_vertices:
    :param radius:
    :param nb_outliers:
    :param ref_graph:
    :param noise_node:
    :param noise_edge:
    :param nb_neighbors_to_consider:
    :return:
    """
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []
    list_ground_truth = []

    graph_index = 0
    while graph_index <= nb_sample_graphs:

        ground_truth, noisy_graph = generate_noisy_graph.run(
            reference_graph,
            nb_vertices,
            noise_node,
            noise_edge
        )

        if not nx.is_connected(noisy_graph):
            continue

        # Add id to edge
        add_integer_id_to_edges(noisy_graph)
        # Save the graph
        list_noisy_graphs.append(noisy_graph)
        # Save all ground-truth for later selecting the selected graphs
        list_ground_truth.append(ground_truth)
        graph_index+=1

    min_geo = []
    selected_graphs = []
    selected_ground_truth = []

    for graphs, gt in zip(list_noisy_graphs, list_ground_truth):
        z = mean_edge_len(graphs)

        if min(z) > 7.0:
            selected_graphs.append(graphs)  # select the noisy graph.
            selected_ground_truth.append(gt)  # and its corresponding ground-truth.
            min_geo.append(min(z))

    sorted_zipped_lists = zip(min_geo, selected_graphs, selected_ground_truth)
    sorted_zipped_lists = sorted(sorted_zipped_lists, reverse=True)

    sorted_graphs = []
    sorted_ground_truth = []

    for l, m, n in sorted_zipped_lists:
        sorted_graphs.append(m)
        sorted_ground_truth.append(n)

    # print("Verifying len of sorted_graphs,sorted_ground_truth,min_geo(should be equal):",
    #       len(sorted_graphs),
    #       len(sorted_ground_truth),
    #       len(min_geo)
    #       )

    ground_truth_perm_to_ref = sorted_ground_truth[:nb_graphs]

    # We generate the ground_truth permutation between graphs
    #print("Groundtruth Labeling..")
    gtp = ground_truth_labeling(ground_truth_perm_to_ref, nb_graphs)

    return sorted_graphs[:nb_graphs], gtp, ground_truth_perm_to_ref
