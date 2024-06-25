import os, sys

from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.utils.graph_processing import get_graph_from_pickle
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
import numpy as np
import networkx as nx
from unittest import TestCase
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


g1 = nx.Graph()
g1.add_node(0, coord=np.array([0, 0]))
g1.add_node(1, coord=np.array([0, 1]))
g1.add_node(2, coord=np.array([0, 2]))
g1.add_edge(0, 1)
g1.add_edge(1, 2)

g2 = nx.Graph()
g2.add_node(0, coord=np.array([1, 0]))
g2.add_node(1, coord=np.array([1, 1]))
g2.add_node(2, coord=np.array([1, 2]))
g2.add_edge(0, 1)
g2.add_edge(1, 2)

g3 = nx.Graph()
g3.add_node(0, coord=np.array([1, 0]))
g3.add_node(1, coord=np.array([1, 1]))
g3.add_node(2, coord=np.array([1, 2]))
g3.add_edge(0, 1)
g3.add_edge(1, 2)

g4 = nx.Graph()
g4.add_node(0, coord=np.array([1, 0]))
g4.add_node(1, coord=np.array([1, 1]))
g4.add_node(2, coord=np.array([1, 2]))
g4.add_node(3, coord=np.array([1, 3]))
g4.add_edge(0, 1)
g4.add_edge(1, 2)
g4.add_edge(2, 3)

g5 = get_graph_from_pickle("test/graph_for_test/graph_00000.gpickle")
g6 = get_graph_from_pickle("test/graph_for_test/graph_00001.gpickle")
graphs = []
for g in os.listdir("test/graph_for_test/"):
    graphs.append(get_graph_from_pickle(os.path.join("test/graph_for_test/", g)))


class TestWassersteinBarycenter(TestCase):
    def test_compute_g1_g2(self):
        b = Barycenter(
            graphs=[g1, g2],
            nb_node=3
        )

        b.compute(fixed_structure=True)
        truth_f = np.array([
            [0.5, 2.0],
            [0.5, 1],
            [0.5, 0]
        ])

        truth_a = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        self.assertTrue(np.array_equal(truth_f, b.F))
        self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute_g1_g2_g3(self):
        b = Barycenter(
            graphs=[g1, g2, g3],
            nb_node=3
        )

        b.compute(fixed_structure=True)

        truth_f = np.array([
            [0.66666667, 2.0],
            [0.66666667, 1.0],
            [0.66666667, 0.0]
        ])

        truth_a = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        self.assertTrue(np.allclose(truth_f, b.F))
        self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute_g1_g2_g3_g4(self):
        b = Barycenter(
            graphs=[g1, g2, g3, g4],
            nb_node=3
        )

        b.compute(fixed_structure=True)

        truth_f = np.array([
            [0.75, 2.1875],
            [0.75, 1.125],
            [0.75, 0.0625]
        ])

        truth_a = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        self.assertTrue(np.allclose(truth_f, b.F))
        self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute(self):
        b = Barycenter(
            graphs=graphs,
            nb_node=30
        )

        b.compute()

        v = Visualisation(title="barycenter" ,graph=b.get_graph(), sphere_radius=90)
        v.construct_sphere()

        v.plot_graphs(folder_path="test/graph_for_test/", radius=90)
        #v.save_as_html("./../algorithms/mean/")
