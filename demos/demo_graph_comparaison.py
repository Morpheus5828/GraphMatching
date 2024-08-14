"""This module plot bars to compare node distance between graphs
..moduleauthor:: Marius Thorre

"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.utils.graph_processing import get_graph_from_pickle, get_distance_between_graphs
from graph_matching.algorithms.barycenter.fgw_barycenter import Barycenter


# graph folder with no outliers detected
path_folder = os.path.join(project_path, "resources/graph_for_test/generation/without_outliers")

dist = []
# compute distance between each family noise barycenter and their graphs used for her generation
for folder in reversed(os.listdir(path_folder)):
    graphs = []
    for graph in os.listdir(os.path.join(path_folder, folder)):
        graphs.append(get_graph_from_pickle(os.path.join(path_folder, folder, graph)))
    b = Barycenter(
        graphs=graphs,
        nb_node=30
    )

    b.compute()
    bary = b.get_graph()

    distances = get_distance_between_graphs(first_graph=bary, graphs=graphs)
    dist.append(sum(distances.values()) / len(distances.values()))

# 'without outliers' graphs folder contain 60 different graph families
plt.plot(np.arange(60), dist)
plt.xlabel("Noise")
plt.ylabel("Distance")
plt.title("Distance between \n Barycenter and graph generation without outliers")
plt.show()

