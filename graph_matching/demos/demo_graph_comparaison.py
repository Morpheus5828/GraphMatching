"""This module plot bars to compare node distance between graphs
..moduleauthor:: Marius Thorre

"""
import os, sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_matching.utils.graph_processing import get_graph_from_pickle, get_distance_between_graphs
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
import graph_matching.algorithms.mean.fugw_barycenter as fugw_barycenter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

path_folder = os.path.join(project_path, "resources/graph_for_test/generation/without_outliers")

dist = []
a = 0
for folder in reversed(os.listdir(path_folder)):
    if a <= 2:
        graphs = []
        for graph in os.listdir(os.path.join(path_folder, folder)):
            graphs.append(get_graph_from_pickle(os.path.join(path_folder, folder, graph)))
        # b = Barycenter(
        #     graphs=graphs,
        #     nb_node=30
        # )
        # b.compute()
        # bary = b.get_graph()

        F_b, _ = fugw_barycenter.compute(
            graphs=graphs,
            rho=1,
            epsilon=0.01,
            alpha=0.15
        )

        g2 = nx.Graph()

        for i, coord in enumerate(F_b):
            g2.add_node(i, coord=coord, label=i)

        distances = get_distance_between_graphs(first_graph=g2, graphs=graphs)
        dist.append(sum(distances.values()) / len(distances.values()))
        a += 1
print(dist)
plt.plot(np.arange(2), dist)
plt.xlabel("Noise")
#plt.text(10, 100, "Max outliers: 10")
plt.ylabel("Distance")
plt.title("Distance between \n Barycenter and graph generation without outliers")
plt.show()
#plt.savefig("C:/Users/thorr/OneDrive/Bureau/Stage/Distance_between_Barycenter_and_graph_generation_without_outliers_fibo")


