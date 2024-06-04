import networkx as nx
from graph_matching.utils.graph.graph_tools import *
from graph_matching.graph_generation.generate_nb_outliers_and_nb_supress import _generate_nb_outliers_and_nb_supress
import graph_matching.graph_generation.generate_sphere_random_sampling as generate_sphere_random_sampling
from resources.slam import topology

def generate_noisy_graph(
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
    key = []
    value = []

    nb_outliers, nb_supress = _generate_nb_outliers_and_nb_supress(nb_vertices)
    # nb_outliers = 0 # TEMPORARILY
    # nb_supress = 0

    noisy_coord_all = noisy_coord

    # Supress Non-Outlier nodes
    if nb_supress > 0:
        # print("nb_supress : ",nb_supress)

        supress_list = random.sample(range(len(noisy_coord)), nb_supress)  # Indexes to remove
        removed_coords = [noisy_coord[i] for i in range(len(noisy_coord)) if i in supress_list]
        # noisy_coord = [dummy_coords if i in supress_list else noisy_coord[i] for i in range(len(noisy_coord))]
        noisy_coord = [noisy_coord[i] for i in range(len(noisy_coord)) if i not in supress_list]

    # Add Outliers
    sphere_random_sampling = []
    if nb_outliers > 0:

        # print("nb_outliers: ", nb_outliers)

        sphere_random_sampling = generate_sphere_random_sampling.run(vertex_number=nb_outliers, radius=radius)
        # merge pertubated and outlier coordinates to add edges
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

    counter = 0
    check = False

    # for outlier in sphere_random_sampling:
    #     for i in range(len(noisy_graph.nodes)):

    #         if np.linalg.norm(outlier - noisy_graph.nodes[i]['coord']) == 0.:

    #             if i<nb_vertices:
    #                 value.append(i)

    # if nb_outliers > 0 and len(key)!=0:
    #     index = 0
    #     for j in range(len(ground_truth_permutation)):
    #         if ground_truth_permutation[j] == key[index]:
    #             ground_truth_permutation[j] = value[index]
    #             index+=1
    #             if index == len(key):
    #                 break

    #     key = key + value
    #     value = value + key

    #     mapping = dict(zip(key,value))
    #     #print("mapping :",mapping)
    #     #print("number of nodes in graphs: ", len(noisy_graph.nodes))
    #     noisy_graph = nx.relabel_nodes(noisy_graph, mapping)

    # Remove 10% of random edges
    edge_to_remove = edge_len_threshold(noisy_graph, 0.10)
    noisy_graph.remove_edges_from(edge_to_remove)

    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))

    gtp = extract_ground_truth_permutation(noisy_graph, noisy_coord_all, sphere_random_sampling)

    return gtp, noisy_graph