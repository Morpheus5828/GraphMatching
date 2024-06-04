"""This module contains code to generate noisy graph
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

from graph_matching.utils.graph.graph_tools import *
import graph_matching.graph_generation.generate_nb_outliers_and_nb_supress as generate_nb_outliers_and_nb_supress
import graph_matching.graph_generation.generate_sphere_random_sampling as generate_sphere_random_sampling
from resources.slam import topology


def run(
        original_graph: nx.Graph,
        nb_vertices: int,
        sigma_noise_nodes: int = 1,
        sigma_noise_edges: int = 1,
        radius: int = 100
) -> (list, nx.Graph):
    """ Generate a noisy graph
    :param original_graph:
    :param nb_vertices:
    :param sigma_noise_nodes:
    :param sigma_noise_edges:
    :param radius:
    :return:
    """

    noisy_coord = von_mises_sampling(nb_vertices, original_graph, sigma_noise_edges)

    nb_outliers, nb_supress = generate_nb_outliers_and_nb_supress.run(nb_vertices)

    noisy_coord_all = noisy_coord

    # Supress Non-Outlier nodes
    if nb_supress > 0:
        supress_list = random.sample(range(len(noisy_coord)), nb_supress)  # Indexes to remove
        noisy_coord = [noisy_coord[i] for i in range(len(noisy_coord)) if i not in supress_list]

    # Add Outliers
    sphere_random_sampling = []
    if nb_outliers > 0:
        sphere_random_sampling = generate_sphere_random_sampling.run(vertex_number=nb_outliers, radius=radius)
        all_coord = noisy_coord + list(sphere_random_sampling)
    else:
        all_coord = noisy_coord

    compute_noisy_edges = tri_from_hull(all_coord)  # take all peturbated coord and comp conv hull.
    adja = topology.adjacency_matrix(compute_noisy_edges)  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_array(adja.todense())

    node_attribute_dict = {}
    for node in noisy_graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(compute_noisy_edges.vertices[node]), 'is_dummy': False,
                                     'is_outlier': False}

    nx.set_node_attributes(noisy_graph, node_attribute_dict)
    nx.set_edge_attributes(noisy_graph, 1.0, name="weight")
    nx.set_edge_attributes(noisy_graph, compute_edge_attribute(noisy_graph, radius))

    edge_to_remove = edge_len_threshold(noisy_graph, 0.10)
    noisy_graph.remove_edges_from(edge_to_remove)

    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))

    gtp = extract_ground_truth_permutation(noisy_graph, noisy_coord_all, sphere_random_sampling)

    return gtp, noisy_graph