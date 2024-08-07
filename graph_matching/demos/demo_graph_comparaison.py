"""This module plot bars to compare node distance between graphs
..moduleauthor:: Marius Thorre

"""
import os, sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.utils.graph_processing import get_graph_from_pickle, get_distance_between_graphs
import graph_matching.algorithms.mean.fugw_barycenter as fugw_barycenter

path_folder = os.path.join(project_path, "resources/graph_for_test/generation/without_outliers")

dist = []
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

    g2 = nx.Graph()

    for i, coord in enumerate(bary):
        g2.add_node(i, coord=coord, label=i)

    distances = get_distance_between_graphs(first_graph=g2, graphs=graphs)
    dist.append(sum(distances.values()) / len(distances.values()))

print(dist)
plt.plot(np.arange(2), dist)
plt.xlabel("Noise")
#plt.text(10, 100, "Max outliers: 10")
plt.ylabel("Distance")
plt.title("Distance between \n Barycenter and graph generation without outliers")
plt.show()
#plt.savefig("C:/Users/thorr/OneDrive/Bureau/Stage/Distance_between_Barycenter_and_graph_generation_without_outliers_fibo")


