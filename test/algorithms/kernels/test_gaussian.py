from unittest import TestCase

import networkx as nx
import sys, os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
import graph_matching.algorithms.kernels.gaussian as gaussian


class TestGaussian(TestCase):
    def test_create_gaussian_node_kernel(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=2.0)
        graph1.add_edge(0, 1, weight=1.0)

        gaussian_kernel = gaussian.create_gaussian_node_kernel(1.0, "weight")

        self.assertEqual(gaussian_kernel(graph1, 0, graph1, 0), 1.0)
