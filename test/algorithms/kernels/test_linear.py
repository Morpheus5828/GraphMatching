from unittest import TestCase

import networkx as nx

import graph_matching.algorithms.kernels.linear as linear
import sys, os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

class TestLinear(TestCase):
    def test_create_linear_node_kernel(self):
        graph1 = nx.Graph()
        graph1.add_node(0, weight=1.0)
        graph1.add_node(1, weight=2.0)
        graph1.add_edge(0, 1, weight=1.0)

        linear_kernel = linear.create_linear_node_kernel("weight")

        self.assertEqual(linear_kernel(graph1, 0, graph1, 1), 2.0)
