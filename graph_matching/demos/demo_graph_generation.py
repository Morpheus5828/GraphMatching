"""Example of graph generation
.. moduleauthor:: Marius Thorre
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from graph_matching.graph_generation.generation_graph_edge_permutation import EdgePermutation


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
    nb_neighbors_to_consider_outliers = 10

    ep = EdgePermutation(
        nb_runs=nb_runs,
        nb_sample_graphs=nb_sample_graphs,
        nb_graphs=nb_graphs,
        nb_vertices=nb_vertices,
        min_noise=min_noise,
        max_noise=max_noise,
        step_noise=step_noise,
        max_outliers=max_outliers,
        step_outliers=step_outliers,
        save_reference=save_reference,
        nb_ref_graph=nb_ref_graph,
        radius=radius,
        nb_neighbors_to_consider_outliers=nb_neighbors_to_consider_outliers,
        generation_folder_path=generation_folder_path
    )
