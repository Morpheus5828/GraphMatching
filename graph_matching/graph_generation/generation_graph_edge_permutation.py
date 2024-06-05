"""This module contains code to generate graph edge using permutation
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import os
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from graph_matching.utils.pickle import save_figure
#from graph_matching.utils.graph.display_graph_tools import Visualisation
import graph_matching.graph_generation.generate_reference_graph as generate_reference_graph
import graph_matching.graph_generation.generate_graph_family as generate_graph_family
from graph_matching.utils.graph.graph_tools import mean_edge_len


class EdgePermutation:
    def __init__(
            self,
            title: str,
            nb_sample_graphs: int,
            nb_graphs: int,
            nb_vertices: int,
            min_noise: int,
            max_noise: int,
            step_noise: int,
            max_outliers: int,
            step_outliers: int,
            save_reference: int,
            nb_ref_graph: int,
            radius: float,
            nb_neighbors_to_consider_outliers: int,
            generation_folder_path: str
    ):
        """
        Compute edge permutation graphs
        :param title:
        :param nb_sample_graphs:
        :param nb_graphs:
        :param nb_vertices:
        :param min_noise:
        :param max_noise:
        :param step_noise:
        :param max_outliers:
        :param step_outliers:
        :param save_reference:
        :param nb_ref_graph:
        :param radius:
        :param nb_neighbors_to_consider_outliers:
        :param generation_folder_path:
        """
        self.title = title
        self.nb_sample_graphs = nb_sample_graphs
        self.nb_graphs = nb_graphs
        self.nb_vertices = nb_vertices
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.step_noise = step_noise
        self.max_outliers = max_outliers
        self.step_outliers = step_outliers
        self.save_reference = save_reference
        self.nb_ref_graph = nb_ref_graph
        self.radius = radius
        self.nb_neighbors_to_consider_outliers = nb_neighbors_to_consider_outliers

        self.path_to_write = generation_folder_path

        trial_path, reference_graph_max = self._generate_families_graph()
        self._generate_noise_graph(
            trial_path=trial_path,
            reference_graph_max=reference_graph_max,
        )

    def _generate_families_graph(self):
        if not os.path.isdir(self.path_to_write):
            os.mkdir(self.path_to_write)


        print("Generating reference_graph..")
        for i in tqdm(range(self.nb_ref_graph)):
            reference_graph = generate_reference_graph.run(self.nb_vertices, self.radius)
            all_geo = mean_edge_len(reference_graph)
            if i == 0:
                min_geo = min(all_geo)
            else:
                if min(all_geo) > min_geo:
                    min_geo = min(all_geo)
                    reference_graph_max = reference_graph
                else:
                    pass
        if self.save_reference:
            print("Selected reference graph with min_geo: ", min_geo)
            trial_path = os.path.join(self.path_to_write, self.title)
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
        save_figure._as_gpickle(
            path=os.path.join(trial_path, "reference_" + self.title),
            graph=reference_graph_max
        )


        return trial_path, reference_graph_max

    def _generate_noise_graph(
            self,
            trial_path: str,
            reference_graph_max: nx.Graph,
    ):

        list_noise = np.arange(self.min_noise, self.max_noise, self.step_noise)

        for noise in list_noise:
            folder_name = f"noise_{noise}_outliers_varied"
            path_parameters_folder = os.path.join(trial_path, folder_name)

            if not os.path.exists(path_parameters_folder):
                os.makedirs(path_parameters_folder)

            list_graphs, ground_truth_perm, ground_truth_perm_to_ref = generate_graph_family.run(
                nb_sample_graphs=self.nb_sample_graphs,
                nb_graphs=self.nb_graphs,
                nb_vertices=self.nb_vertices,
                radius=self.radius,
                nb_outliers=self.max_outliers,
                ref_graph=reference_graph_max,
                noise_node=noise,
                noise_edge=noise,
                nb_neighbors_to_consider=self.nb_neighbors_to_consider_outliers
            )

            for i_family, graph_family in enumerate(list_graphs):
                sorted_graph = nx.Graph()
                sorted_graph.add_nodes_from(sorted(graph_family.nodes(data=True)))
                sorted_graph.add_edges_from(graph_family.edges(data=True))

                save_figure._as_gpickle(
                    path=os.path.join(path_parameters_folder, f"graph_{i_family:05d}"),
                    graph=sorted_graph
                )

