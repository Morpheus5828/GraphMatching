"""This module contains tools to generate barycenter graph from graph list
..Moduleauthor:: Marius Thorre
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
from graph_matching.utils.graph_processing import get_graph_from_pickle

graph_test_path = os.path.join(project_path, "resources/graph_for_test")
graphs = []
for g in os.listdir(os.path.join(graph_test_path, "generation")):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, "generation", g)))

path = "graph_generated/pickle/noise_1810_outliers_varied"



b = Barycenter(
    graphs=graphs,
    nb_node=30
)
b.compute(fixed_structure=True)
bary = b.get_graph()

v = Visualisation(
    graph=bary,
    sphere_radius=90,
    title="barycenter"
)

v.construct_sphere()
v.plot_graphs(folder_path=path)

v.save_as_pickle("graph_generated")
#v.save_as_html("graph_generated")





