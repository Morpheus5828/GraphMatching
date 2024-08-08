"""Example of graph generation
..moduleauthor:: Marius Thorre
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from graph_matching.algorithms.graph_generation.generation_graph_edge_permutation import Graph_Generation

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == '__main__':
    generation_folder_path = os.path.join(current_dir, "graph_generated")
    nb_sample_graphs = 20 # nb graph per noise families
    nb_vertices = 30  # nb nodes that we want to have in graphs
    noise = [1, 1, 61]  # Min noise, Step noise, Max noise
    max_outliers = 10
    step_outliers = 10
    save_reference = 1  # Nb reference we want to save
    nb_ref_graph = 1000  # Nb reference graphs
    radius = 100  # sphere radius, when graph spatial structure will be converted
    nb_neighbors_to_consider_outliers = 10
    print("Starting generation ... ")

    ep = Graph_Generation(
        pickle_folder_title="pickle",  # format to save graph
        html_folder_title="html",  # format to see graph
        nb_sample_graphs=nb_sample_graphs,
        nb_vertices=nb_vertices,
        noise=noise,
        max_outliers=max_outliers,
        step_outliers=step_outliers,
        save_reference=save_reference,
        nb_ref_graph=nb_ref_graph,
        radius=radius,
        nb_neighbors_to_consider_outliers=nb_neighbors_to_consider_outliers,
        generation_folder_path=generation_folder_path
    )

    print(f"Graphs save in {generation_folder_path}")
