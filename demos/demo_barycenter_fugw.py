import os, sys
import concurrent.futures
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from unittest import TestCase

import numpy as np
import networkx as nx
import graph_matching.algorithms.barycenter.fugw_barycenter as fugw_barycenter
from graph_matching.utils.graph_processing import get_graph_coord, get_graph_from_pickle
from graph_matching.utils.display_graph_tools import Visualisation

graph_test_path = os.path.join(
    project_root,
    "resources/graph_for_test/generation/without_outliers/noise_50")

# extract pickle graph file in networkx graph
graphs = [get_graph_from_pickle(os.path.join(graph_test_path, g)) for g in os.listdir(graph_test_path)]

# list which contains different hyperparameter values
rho_values = [1.4, 1.6, 1.8, 2, 2, 2]
alpha_values = [0.29, 0.30, 0.33, 0.35, 0.40, 0.45, 0.47, 0.48]
epsilon_values = [1e-2]
# process time can be more than 2 hours
print("Starting process, please hold on ...")
for epsilon in epsilon_values:
    for rho in rho_values:
        for alpha in alpha_values:
            print(f"Parameter used: rho: {rho}, alpha: {alpha}, epsilon: {epsilon} ")
            # F_b contain nodes coordinates barycenter in 3 dimensions
            F_b = fugw_barycenter.get_graph(graphs, rho, epsilon, alpha)
            # just print 4 first lines
            print(F_b[:4])
            print("####")