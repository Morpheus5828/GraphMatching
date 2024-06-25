import os, sys

from unittest import TestCase

import numpy as np
import networkx as nx

from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
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


class TestWassersteinBarycenter(TestCase):
    def test_compute(self):
        b = Barycenter(
            graphs=[g1, g2],
            nb_node=3
        )

        b.compute()
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

        b = Barycenter(
            graphs=[g1, g2, g3],
            nb_node=3
        )

        b.compute()

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

        b = Barycenter(
            graphs=[g1, g2, g3, g4],
            nb_node=3
        )

        b.compute()

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



