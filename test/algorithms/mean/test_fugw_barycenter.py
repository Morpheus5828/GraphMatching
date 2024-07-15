"""This module test FUGW_barycenter implemntation

.. moduleauthor:: Marius Thorre
"""

import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from unittest import TestCase
import numpy as np
import networkx as nx
import graph_matching.algorithms.mean.fugw_barycenter as fugw_barycenter
from graph_matching.utils.graph_processing import get_graph_coord, get_graph_from_pickle
from graph_matching.utils.display_graph_tools import Visualisation

g1 = nx.Graph()
g1.add_node(0, coord=np.array([25, -12, 15]), label=0)
g1.add_node(1, coord=np.array([-32, 45, -20]), label=1)
g1.add_node(2, coord=np.array([23, -40, 10]), label=2)
g1.add_node(3, coord=np.array([-10, 35, -5]), label=3)
g1.add_node(4, coord=np.array([5, 20, 25]), label=4)
g1.add_node(5, coord=np.array([-45, -25, 30]), label=5)
g1.add_node(6, coord=np.array([15, -30, -15]), label=6)
g1.add_node(7, coord=np.array([10, 10, -25]), label=7)
g1.add_node(8, coord=np.array([-20, 25, 35]), label=8)
g1.add_node(9, coord=np.array([30, -5, -10]), label=9)
g1.add_node(10, coord=np.array([12, -8, 20]), label=10)
g1.add_node(11, coord=np.array([-20, 40, -10]), label=11)
g1.add_node(12, coord=np.array([10, -45, 15]), label=12)
g1.add_node(13, coord=np.array([-15, 30, -5]), label=13)
g1.add_node(14, coord=np.array([0, 15, 25]), label=14)
g1.add_node(15, coord=np.array([-35, -20, 10]), label=15)
g1.add_node(16, coord=np.array([20, -25, -20]), label=16)
g1.add_node(17, coord=np.array([5, 5, 30]), label=17)
g1.add_node(18, coord=np.array([-25, 20, -15]), label=18)
g1.add_node(19, coord=np.array([35, -10, 5]), label=19)
g1.add_node(20, coord=np.array([25, -12, 5]), label=20)
g1.add_node(21, coord=np.array([-32, 45, 15]), label=21)
g1.add_node(22, coord=np.array([23, -40, -25]), label=22)
g1.add_node(23, coord=np.array([-10, 35, 20]), label=23)
g1.add_node(24, coord=np.array([5, 20, -10]), label=24)
g1.add_node(25, coord=np.array([-45, -25, 15]), label=25)
g1.add_node(26, coord=np.array([15, -30, 25]), label=26)
g1.add_node(27, coord=np.array([10, 10, -5]), label=27)
g1.add_node(28, coord=np.array([-20, 25, 10]), label=28)
g1.add_node(29, coord=np.array([30, -5, -15]), label=29)

edges_g1 = [(0, 2), (1, 3), (4, 7), (5, 6), (8, 9), (0, 6), (2, 8), (1, 7), (3, 5), (4, 9),
            (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (10, 12), (11, 13), (14, 16), (15, 17), (18, 10),
            (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (20, 22), (21, 23), (24, 26), (25, 27), (28, 20)]
for edge in edges_g1:
    g1.add_edge(*edge)

g2 = nx.Graph()
g2.add_node(0, coord=np.array([12, -8, 20]), label=0)
g2.add_node(1, coord=np.array([-20, 40, -10]), label=1)
g2.add_node(2, coord=np.array([10, -45, 15]), label=2)
g2.add_node(3, coord=np.array([-15, 30, -5]), label=3)
g2.add_node(4, coord=np.array([0, 15, 25]), label=4)
g2.add_node(5, coord=np.array([-35, -20, 10]), label=5)
g2.add_node(6, coord=np.array([20, -25, -20]), label=6)
g2.add_node(7, coord=np.array([5, 5, 30]), label=7)
g2.add_node(8, coord=np.array([-25, 20, -15]), label=8)
g2.add_node(9, coord=np.array([35, -10, 5]), label=9)
g2.add_node(10, coord=np.array([25, -15, 35]), label=10)
g2.add_node(11, coord=np.array([-22, 33, -12]), label=11)
g2.add_node(12, coord=np.array([17, -42, 18]), label=12)
g2.add_node(13, coord=np.array([-18, 27, -8]), label=13)
g2.add_node(14, coord=np.array([3, 12, 28]), label=14)
g2.add_node(15, coord=np.array([-38, -18, 14]), label=15)
g2.add_node(16, coord=np.array([22, -28, -22]), label=16)
g2.add_node(17, coord=np.array([8, 8, 33]), label=17)
g2.add_node(18, coord=np.array([-28, 22, -18]), label=18)
g2.add_node(19, coord=np.array([38, -12, 8]), label=19)
g2.add_node(20, coord=np.array([20, -10, 30]), label=20)
g2.add_node(21, coord=np.array([-15, 35, -25]), label=21)
g2.add_node(22, coord=np.array([18, -48, 22]), label=22)
g2.add_node(23, coord=np.array([-20, 25, -5]), label=23)
g2.add_node(24, coord=np.array([5, 18, 20]), label=24)
g2.add_node(25, coord=np.array([-30, -15, 8]), label=25)
g2.add_node(26, coord=np.array([25, -30, -18]), label=26)
g2.add_node(27, coord=np.array([12, 12, 35]), label=27)
g2.add_node(28, coord=np.array([-22, 28, -12]), label=28)
g2.add_node(29, coord=np.array([30, -8, 10]), label=29)

edges_g2 = [(0, 3), (1, 4), (2, 6), (3, 8), (4, 7), (5, 9), (0, 5), (1, 6), (2, 9), (3, 7),
            (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (10, 12), (11, 13), (14, 16), (15, 17), (18, 10),
            (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (20, 22), (21, 23), (24, 26), (25, 27), (28, 20)]
for edge in edges_g2:
    g2.add_edge(*edge)

G0 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_01",
        "graph_00000.gpickle"
    )
)

G10 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_01",
        "graph_00010.gpickle"
    )
)

G19 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_01",
        "graph_00019.gpickle"
    )
)

graph_test_path = os.path.join(
    project_root,
    "resources/graph_for_test/generation/without_outliers/noise_60"
)
graphs = []
for g in os.listdir(graph_test_path):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))


class TestFUGW_barycenter(TestCase):
    def test_compute(self):
        F_b, D_b = fugw_barycenter.compute(graphs=[g1, g2])
        print(F_b, D_b)

        # tmp = nx.Graph()
        # for node, i in enumerate(F_b):
        #     tmp.add_node(node, coord=i*10e7, label=0)
        #     print(node, i)
        # v = Visualisation(
        #     graph=tmp,
        #     sphere_radius=100,
        #     title="181 without outliers"
        # )
        # v.plot_graphs(folder_path=graph_test_path)

    def test_get_init_graph(self):
        result = fugw_barycenter._get_init_graph(graphs)
        self.assertTrue(result.shape == (30, 3))
