"""This module contains code to generate Graph

..moduleauthor:: Marius Thorre, Rohit Yadav
"""
import warnings

warnings.filterwarnings("ignore")
import sys
import os
sys.path.append("resources/slam")
sys.path.append("graph_matching/utils")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_matching.utils.graph_tools import *
from resources.slam import topology as stop
from graph_matching.utils.graph_processing import *

import pickle
import os
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import betabinom
import random


def generate_reference_graph(
        nb_vertices: int,
        radius: float
) -> nx.Graph:
    """ Generate random sampling
    :param int nb_vertices:
    :param float radius:
    :return nx.Graph :
    :rtype: np.ndarray
    """
    sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_vertices, radius=radius)
    sphere_random_sampling = tri_from_hull(sphere_random_sampling)  # Computing convex hull (adding edges)

    adja = stop.adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_array(adja.todense())
    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(sphere_random_sampling.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)

    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in graph.edges:
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = graph_processing.get_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)

    return graph


def generate_sphere_random_sampling(
        vertex_number: int = 100,
        radius: float = 1.0
) -> np.ndarray:
    """ Generate a sphere with random sampling
    :param vertex_number:
    :param radius:
    :return : sphere coordinate array
    :rtype : np.ndarray
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return coords


def generate_nb_outliers_and_nb_supress(
        nb_vertices: int
) -> tuple:
    """ Sample nb_outliers and nb_supress from a Normal distance following the std of real data
    :param nb_vertices:
    :return: Tuple which contains nb outliers and nb supress
    :rtype: (int, int)
    """
    # mean_real_data = 40         # mean real data
    std_real_data = 4  # std real data

    mu = 10  # mu_A = mu_B = mu
    sigma = std_real_data
    n = 25

    alpha = compute_alpha(n, mu, sigma ** 2)  # corresponding alpha with respect to given mu and sigma
    beta = compute_beta(alpha, n, mu)  # corresponding beta

    nb_supress = betabinom.rvs(n, alpha, beta, size=1)[0]
    nb_outliers = betabinom.rvs(n, alpha, beta, size=1)[0]  # Sample nb_outliers

    return int(nb_outliers), int(nb_supress)


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

    nb_outliers, nb_supress = generate_nb_outliers_and_nb_supress(nb_vertices)
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

        sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_outliers, radius=radius)
        # merge pertubated and outlier coordinates to add edges
        all_coord = noisy_coord + list(sphere_random_sampling)
    else:
        all_coord = noisy_coord

    compute_noisy_edges = tri_from_hull(all_coord)  # take all peturbated coord and comp conv hull.
    adja = stop.adjacency_matrix(compute_noisy_edges)  # compute the new adjacency mat.

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


def generate_graph_family(
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

    # We generate the n noisy graphs
    print("Generating graphs..")

    for _ in tqdm(range(nb_sample_graphs)):

        ground_truth, noisy_graph = generate_noisy_graph(
            reference_graph,
            nb_vertices,
            noise_node,
            noise_edge
        )
        # Add outliers
        # add_outliers(noisy_graph, nb_outliers, nb_neighbors_to_consider, radius)

        if nx.is_connected(noisy_graph) == False:
            continue

        if nx.is_connected(noisy_graph) == False:
            print("Found disconnected components..!!")

        # Add id to edge
        add_integer_id_to_edges(noisy_graph)

        # Save the graph
        list_noisy_graphs.append(noisy_graph)

        # Save all ground-truth for later selecting the selected graphs
        list_ground_truth.append(ground_truth)

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

    print("Verifying len of sorted_graphs,sorted_ground_truth,min_geo(should be equal):",
          len(sorted_graphs),
          len(sorted_ground_truth),
          len(min_geo)
          )

    ground_truth_perm_to_ref = sorted_ground_truth[:nb_graphs]

    # We generate the ground_truth permutation between graphs
    print("Groundtruth Labeling..")
    gtp = ground_truth_labeling(ground_truth_perm_to_ref, nb_graphs)

    return sorted_graphs[:nb_graphs], gtp, ground_truth_perm_to_ref


def generate_n_graph_family_and_save(
        path_to_write: str,
        nb_runs: int,
        nb_ref_graph: int,
        nb_sample_graphs: int,
        nb_graphs: int,
        nb_vertices: int,
        radius: float,
        list_noise: np.ndarray,
        max_outliers: int,
        nb_neighbors_to_consider: int = 10,
        save_reference=0
):
    """ Generate n family of graphs for each couple (noise, outliers).
    The graphs are saved in a folder structure at the point path_to_write
    :param path_to_write:
    :param nb_runs:
    :param nb_ref_graph:
    :param nb_sample_graphs:
    :param nb_graphs:
    :param nb_vertices:
    :param radius:
    :param list_noise:
    :param max_outliers:
    :param nb_neighbors_to_consider:
    :param save_reference:
    :return:
    """

    # check if the path given is a folder otherwise create one
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    # generate n families of graphs
    for i_graph in range(nb_runs):

        # Select the ref graph with the highest mean geo distance
        print("Generating reference_graph..")

        for i in tqdm(range(nb_ref_graph)):
            reference_graph = generate_reference_graph(nb_vertices, radius)
            all_geo = mean_edge_len(reference_graph)

            if i == 0:
                min_geo = min(all_geo)

            else:
                if min(all_geo) > min_geo:
                    min_geo = min(all_geo)
                    reference_graph_max = reference_graph

                else:
                    pass

        if save_reference:
            print("Selected reference graph with min_geo: ", min_geo)
            trial_path = os.path.join(path_to_write, str(i_graph))  # for each trial
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
        with open(os.path.join(trial_path, "reference_" + str(i_graph) + ".gpickle"), "wb") as f:
            pickle.dump(reference_graph_max, f, pickle.HIGHEST_PROTOCOL)

        # nx.write_gpickle(reference_graph_max, os.path.join(trial_path, "reference_" + str(i_graph) + ".gpickle"))

    for noise in list_noise:
        # for outliers in list_outliers:

        folder_name = "noise_" + str(noise) + ",outliers_varied"  # + str(max_outliers)
        path_parameters_folder = os.path.join(trial_path, folder_name)

        if not os.path.isdir(path_parameters_folder):
            os.mkdir(path_parameters_folder)
            os.mkdir(os.path.join(path_parameters_folder, "graphs"))

        list_graphs, ground_truth_perm, ground_truth_perm_to_ref = generate_graph_family(
            nb_sample_graphs=nb_sample_graphs, nb_graphs=nb_graphs,
            nb_vertices=nb_vertices,
            radius=radius,
            nb_outliers=max_outliers,
            ref_graph=reference_graph_max,
            noise_node=noise,
            noise_edge=noise,
            nb_neighbors_to_consider=nb_neighbors_to_consider)

        for i_family, graph_family in enumerate(list_graphs):
            sorted_graph = nx.Graph()
            sorted_graph.add_nodes_from(sorted(graph_family.nodes(data=True)))  # Sort the nodes of the graph by key
            sorted_graph.add_edges_from(graph_family.edges(data=True))

            print("Length of noisy graph: ", len(sorted_graph.nodes))

            # nx.draw(sorted_graph)
            # with open("graph_{:05d}.format(i_family)" + "g.pickle", "wb") as f:
            #     pickle.dump(sorted_graph, f, pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(path_parameters_folder, "graphs","graph_{:05d}".format(i_family) + ".gpickle"), "wb") as f:
                pickle.dump(sorted_graph, f, pickle.HIGHEST_PROTOCOL)


        # np.save(os.path.join(path_parameters_folder, "ground_truth"), ground_truth_perm)
        #
        # with open(path_parameters_folder + "/permutation_to_ref_graph.gpickle", 'wb') as f:
        #     pickle.dump(ground_truth_perm_to_ref, f, pickle.HIGHEST_PROTOCOL)
        #
        # with open(path_parameters_folder + "/ground_truth.gpickle", 'wb') as f:
        #     pickle.dump(ground_truth_perm, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    path_to_write = 'graph_generated'

    nb_runs = 4
    nb_sample_graphs = 1000  # # of graphs to generate before selecting the NN graphs with highest geodesic distance.
    nb_graphs = 20  # 134 # nb of graphs to generate
    nb_vertices = 30  # 88 as per real data mean  #72 based on Kaltenmark, MEDIA, 2020 // 88 based on the avg number of nodes in the real data.
    min_noise = 100
    max_noise = 1400
    step_noise = 300
    # min_outliers = 0
    max_outliers = 20
    step_outliers = 10
    save_reference = 1
    nb_ref_graph = 1000
    radius = 100

    list_noise = np.arange(min_noise, max_noise, step_noise)
    # list_outliers = np.array(list(range(min_outliers, max_outliers, step_outliers)))
    nb_neighbors_to_consider_outliers = 10

    # call the generation procedure
    generate_n_graph_family_and_save(
        path_to_write=path_to_write,
        nb_runs=nb_runs,
        nb_ref_graph=nb_ref_graph,
        nb_sample_graphs=nb_sample_graphs,
        nb_graphs=nb_graphs,
        nb_vertices=nb_vertices,
        radius=radius,
        list_noise=list_noise,
        max_outliers=max_outliers,
        nb_neighbors_to_consider=nb_neighbors_to_consider_outliers,
        save_reference=save_reference
    )
