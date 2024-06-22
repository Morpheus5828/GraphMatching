"""This module contains code to generate graph edge using permutation
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import os, sys
import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from graph_matching.utils.display_graph_tools import Visualisation
import graph_matching.algorithms.graph_generation.generate_reference_graph as generate_reference_graph
import graph_matching.algorithms.graph_generation.generate_graph_family as generate_graph_family
from graph_matching.utils.graph_tools import mean_edge_len

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


class EdgePermutation:
    def __init__(
            self,
            pickle_folder_title: str,
            nb_sample_graphs: int,
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
            generation_folder_path: str,
            html_folder_title: str = None
    ):
        """
        Compute edge permutation graphs
        :param pickle_folder_title:
        :param nb_sample_graphs:
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
        self.pickle_folder_title = pickle_folder_title
        self.nb_sample_graphs = nb_sample_graphs
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
        self.html_folder_title = html_folder_title
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
            trial_path = os.path.join(self.path_to_write, self.pickle_folder_title)
            html_path = os.path.join(self.path_to_write, self.html_folder_title)
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
            if not os.path.isdir(html_path):
                os.mkdir(html_path)

        v = Visualisation(
            graph=reference_graph,
            title="reference"
        )

        v.save_as_pickle(path_to_save=trial_path)
        v.save_as_html(path_to_save=os.path.join(html_path))
        return trial_path, reference_graph_max

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
            trial_path = os.path.join(self.path_to_write, self.pickle_folder_title)
            html_path = os.path.join(self.path_to_write, self.html_folder_title)
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
            if not os.path.isdir(html_path):
                os.mkdir(html_path)

        v = Visualisation(
            graph=reference_graph,
            title="reference",
            sphere_radius=self.radius
        )

        v.save_as_pickle(path_to_save=trial_path)
        v.save_as_html(path_to_save=os.path.join(os.path.join(project_path, self.path_to_write, self.html_folder_title)))
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
                nb_vertices=self.nb_vertices,
                ref_graph=reference_graph_max,
                noise_node=noise,
                noise_edge=noise,
            )

            for i_family, graph_family in enumerate(list_graphs):
                sorted_graph = nx.Graph()
                sorted_graph.add_nodes_from(sorted(graph_family.nodes(data=True)))
                sorted_graph.add_edges_from(graph_family.edges(data=True))
                if not os.path.exists(os.path.join(
                        project_path + self.html_folder_title,
                        folder_name)):
                    os.makedirs(os.path.join(
                        project_path + self.html_folder_title,
                        folder_name))
                v = Visualisation(graph=sorted_graph, sphere_radius=self.radius, title=f"graph_{i_family:05d}")
                v.save_as_html(os.path.join(project_path, self.path_to_write,self.html_folder_title, folder_name))
                v.save_as_pickle(os.path.join(project_path, self.path_to_write,self.pickle_folder_title, folder_name))


