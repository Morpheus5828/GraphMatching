"""This module test FUGW_barycenter implemntation

.. moduleauthor:: Marius Thorre
"""

import os, sys
import concurrent.futures
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
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
    "resources/graph_for_test/generation/without_outliers/noise_60"
)
graphs = []
for g in os.listdir(graph_test_path):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))


class TestFUGW_barycenter(TestCase):

    def test_compute(self):
        F_b, _ = fugw_barycenter.compute(
            graphs=graphs,
            rho=1,
            epsilon=0.01,
            alpha=0.35
        )

        # tmp = nx.Graph()
        # for node, i in enumerate(F_b):
        #     tmp.add_node(node, coord=i, label=0)
        #     print(node, i)
        #
        # v = Visualisation(
        #     graph=tmp,
        #     sphere_radius=100,
        #     title="181 without outliers"
        # )
        # v.plot_graphs(folder_path=graph_test_path)
