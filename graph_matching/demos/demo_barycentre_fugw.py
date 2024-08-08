import os, sys
import concurrent.futures
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from unittest import TestCase

import numpy as np
import networkx as nx
import graph_matching.algorithms.mean.fugw_barycenter as fugw_barycenter
from graph_matching.utils.graph_processing import get_graph_coord, get_graph_from_pickle
from graph_matching.utils.display_graph_tools import Visualisation

graph_test_path = os.path.join(
    project_root,
    "resources/graph_for_test/generation/without_outliers/noise_55")

# extract pickle graph file in networkx graph
graphs = [get_graph_from_pickle(os.path.join(folder_path, g)) for g in os.listdir(folder_path)]

# list which contains different value
rho_values = [1, 1.2]
alpha_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
epsilon_values = [1e-2, 1e-3, 1e-4]

for epsilon in epsilon_values:
    for rho in rho_values:
        for alpha in alpha_values:
            print(rho, alpha, epsilon)
            F_b = fugw_barycenter.get_graph(graphs, rho, epsilon, alpha)
            print(F_b[:4])
            print("####")