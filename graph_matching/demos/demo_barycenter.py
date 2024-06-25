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

script_dir = os.path.dirname(os.path.abspath(__file__))
graph_folder = os.path.join(script_dir, "graph_generated", "pickle")

graphs = []
path = "graph_generated/pickle/noise_1810_outliers_varied"

for graph in os.listdir(path):
    g = get_graph_from_pickle(os.path.join(path, graph))
    graphs.append(g)


barycenter = Barycenter(
    graphs=graphs,
    nb_node=25,
    find_tresh_inf=0.5,
    find_tresh_sup=100,
    find_tresh_step=100,
    graph_vmin=-1,
    graph_vmax=1
)

bary_graph = barycenter.get_graph()
v = Visualisation(
    graph=bary_graph,
    sphere_radius=100,
    title="barycenter"
)

v.construct_sphere()
v.plot_graphs(folder_path=path)
v.save_as_html("graph_generated")





