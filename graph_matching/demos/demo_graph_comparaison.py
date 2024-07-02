"""This module plot bars to compare node distance between graphs
..moduleauthor:: Marius Thorre

"""
import os, sys
import matplotlib.pyplot as plt
from graph_matching.utils.graph_processing import get_graph_from_pickle, get_distance_between_graphs
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

#gref = get_graph_from_pickle(os.path.join(project_path, "GraphMatching/resources/graph_for_test/reference.gpickle"))

graph_test_path = os.path.join(project_path, "resources/graph_for_test/generation/noise_181_outliers_varied")
graphs = []
for g in os.listdir(graph_test_path):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))

b = Barycenter(
    graphs=graphs,
    nb_node=30  # because graphs has 30 nodes
)

b.compute()
bary = b.get_graph()

distances = get_distance_between_graphs(first_graph=bary, graphs=graphs)
distances = dict(sorted(distances.items()))

plt.bar(list(distances.keys()), list(distances.values()))

plt.xlabel("Node label")
plt.ylabel("Distance")
plt.title("Distance between \n Barycenter and graph generation")
plt.savefig("C:/Users/thorr/OneDrive/Bureau/Stage/bary_all_graph181_so")

# plt.clf()
#
# distances = get_distance_between_graphs(first_graph=gref, graphs=graphs)
# distances = dict(sorted(distances.items()))
#
# plt.bar(list(distances.keys()), list(distances.values()))
# plt.xlabel("Node label")
# plt.ylabel("Distance")
# plt.title("Distance between \n Reference and graph generation")
# plt.savefig("C:/Users/thorr/OneDrive/Bureau/Stage/gref_all_graph")

