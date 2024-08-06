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

graph_test_path = os.path.join(project_path, "resources/graph_for_test/")
file_cortex_mesh = os.path.join(project_path, "resources", "template_mesh", "lh.OASIS_testGrp_average_inflated.gii")
file_sphere_mesh = os.path.join(project_path, "resources", "template_mesh", "ico100_7.gii")

folder_path = os.path.join(
    project_path,
    "resources",
    "graph_for_test",
    "generation",
    "without_outliers",
    "noise_60"
)

graphs = []
for g in os.listdir(folder_path):
    graphs.append(get_graph_from_pickle(os.path.join(folder_path, g)))

b = Barycenter(
    graphs=graphs,
    nb_node=30
)


b.compute()

bary = b.get_graph()


v = Visualisation(
    graph=bary,
    sphere_radius=100,
    title="1 without outliers"
)


v.plot_graphs(folder_path=folder_path)
v.show_fig()



