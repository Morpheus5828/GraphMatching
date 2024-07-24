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
        "noise_60",
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
        "noise_60",
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
        ref = np.array([[27.02545743, -96.15696259, 4.84388233],
                        [-34.80607622, -87.40297297, 33.90069872],
                        [-8.62674197, -93.54577754, -34.27487166],
                        [21.81598043, -85.40192331, 47.22895819],
                        [-67.038979, -73.83046696, -7.40523071],
                        [66.09292777, -67.10636595, -33.59256686],
                        [-33.80812468, -56.5852361, 75.20054362],
                        [-38.15003206, -57.85861432, -72.08991471],
                        [89.35082027, -36.44076775, 26.23930951],
                        [-88.00904932, -22.8121901, 41.64146037],
                        [35.22752572, -18.08221777, -91.82622083],
                        [20.16174255, -15.02308706, 96.78745266],
                        [-68.81571439, -26.02998032, -67.72619565],
                        [99.01363249, -9.49484619, -10.30283834],
                        [-56.76721154, -24.11269461, 78.71506624],
                        [-15.10621202, -0.17980351, -98.85226365],
                        [74.97215436, -8.09819117, 65.67796716],
                        [-95.60414461, 28.00876304, -8.68082524],
                        [64.47298002, 19.00291157, -74.04136816],
                        [6.27464211, 39.77670992, 91.53383098],
                        [-47.82885966, 42.9313184, -76.61137047],
                        [86.52953757, 49.02432873, -10.45247911],
                        [-65.20074533, 51.04175325, 56.067836],
                        [23.02375053, 64.51636736, -72.85290148],
                        [60.89560812, 62.55368369, 48.77254933],
                        [-73.44386201, 65.98676111, -15.86652111],
                        [53.40582545, 73.79405745, -41.2583918],
                        [-19.16262371, 83.16697612, 52.11571678],
                        [-9.67947247, 92.18869503, -37.51736025],
                        [29.39696235, 95.25801292, 7.85681732]])
        rho_values = [0.1, 1, 10]
        alpha_values = [0.15, 0.35, 0.50, 0.75]
        epsilon_values = [0.01, 0.001]

        for rho in rho_values:
            for alpha in alpha_values:
                for epsilon in epsilon_values:
                    F_b, D_b = fugw_barycenter.compute(
                        graphs=[G0],
                        rho=rho,
                        epsilon=epsilon,
                        alpha=alpha
                    )
                    F_b *= 100
                    print(rho, alpha, epsilon, np.linalg.norm(ref - F_b))

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

    def test_get_init_graph(self):
        result = fugw_barycenter._get_init_graph(graphs)
        self.assertTrue(result.shape == (30, 3))
