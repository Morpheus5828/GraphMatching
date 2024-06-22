"""This module contains tools to generate barycenter graph from graph list
..Moduleauthor:: Marius Thorre
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import graph_matching.algorithms.mean.wasserstein_barycenter as mean
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.utils.graph_processing import get_graph_from_pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
graph_folder = os.path.join(script_dir, "graph_generated", "pickle")

graphs = []
for file in os.listdir(graph_folder):
    if os.path.isdir(os.path.join(graph_folder, file)):
        for graph in os.listdir(os.path.join(graph_folder, file)):
            g = get_graph_from_pickle(os.path.join(graph_folder, file, graph))
            graphs.append(g)

barycenter = mean.Barycenter(
    graphs=graphs,
    size_bary=30,
    find_tresh_inf=0.5,
    find_tresh_sup=100,
    find_tresh_step=100,
    graph_vmin=-1,
    graph_vmax=1
    )


v = Visualisation(
    graph=barycenter.get_graph(),
    sphere_radius=100,
    title="barycenter"
)

#v.save_as_html(os.path.join(current_dir, "graph_generated"))
v.save_as_pickle(os.path.join(current_dir, "graph_generated"))

# v.compareAndSave(
#     secondGraph=get_graph_from_pickle(os.path.join(project_path, "graph_matching/demos/graph_generated/pickle/reference.gpickle")),
#     path_to_save=current_dir
# )

# reference = get_graph_from_pickle(os.path.join(project_path, "graph_matching/demos/graph_generated/pickle/reference.gpickle"))
# a = np.array([data["coord"] for node, data in barycenter.get_graph().nodes(data=True)])
#
# c = np.array([data["coord"] for node, data in reference.nodes(data=True)])
#
# print(np.linalg.norm(a-c))





