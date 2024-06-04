"""This module contains code to generate Graph

..moduleauthor:: Marius Thorre, Rohit Yadav
"""

import os
import graph_matching.graph_generation.generate_reference_graph as generate_reference_graph
from graph_matching.graph_generation.generate_graph_family import generate_graph_family
import pickle
from tqdm.auto import tqdm
from graph_matching.utils.graph.graph_tools import *
from graph_matching.utils.graph.graph_tools import mean_edge_len


def _save_figure(path: str, graph):
    with open(os.path.join(path + ".gpickle"), "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def _generate_noise_graph(noises: np.ndarray, trial_path: str, reference_graph_max, nb_neighbors_to_consider: int):
    for noise in noises:
        folder_name = "noise_" + str(noise) + ",outliers_varied"
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

            # print("Length of noisy graph: ", len(sorted_graph.nodes))

            #nx.draw(sorted_graph)
            #plt.draw()
            # with open("graph_{:05d}.format(i_family)" + "g.pickle", "wb") as f:
            #     pickle.dump(sorted_graph, f, pickle.HIGHEST_PROTOCOL)

            # with open(os.path.join(path_parameters_folder, "graphs", "graph_{:05d}".format(i_family) + ".gpickle"),
            #           "wb") as f:
            #     pickle.dump(sorted_graph, f, pickle.HIGHEST_PROTOCOL)
            _save_figure(path=os.path.join(path_parameters_folder, "graphs", "graph_{:05d}".format(i_family)),
                         graph=sorted_graph)


def _generate_families_graph(
        path_to_write: str,
        save_reference: int
):
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    for index in range(nb_runs):
        print("Generating reference_graph..")
        for i in tqdm(range(nb_ref_graph)):
            reference_graph = generate_reference_graph.run(nb_vertices, radius)
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
            trial_path = os.path.join(path_to_write, str(index))
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
            trial_path = os.path.join(path_to_write, str(index))
        _save_figure(path=os.path.join(trial_path, "reference_" + str(index)), graph=reference_graph_max)
    return trial_path, reference_graph_max


if __name__ == '__main__':
    generation_folder_path = "graph_generated"

    nb_runs = 4
    nb_sample_graphs = 1000  # # of graphs to generate before selecting the NN graphs with highest geodesic distance.
    nb_graphs = 20  # 134 # nb of graphs to generate
    nb_vertices = 30  # 88 as per real data mean  #72 based on Kaltenmark, MEDIA, 2020 // 88 based on the avg number of nodes in the real data.
    min_noise = 100
    max_noise = 1400
    step_noise = 300
    max_outliers = 20
    step_outliers = 10
    save_reference = 1
    nb_ref_graph = 1000
    radius = 100

    list_noise = np.arange(min_noise, max_noise, step_noise)
    nb_neighbors_to_consider_outliers = 10

    trial_path, reference_graph_max = _generate_families_graph(path_to_write=generation_folder_path, save_reference=save_reference)
    _generate_noise_graph(
        noises=list_noise,
        trial_path=trial_path,
        reference_graph_max=reference_graph_max,
        nb_neighbors_to_consider=nb_neighbors_to_consider_outliers
    )

