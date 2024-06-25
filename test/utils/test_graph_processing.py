"""This module will contain test from graph_processing.py file
..moduleauthor:: Marius Thorre
"""

import networkx as nx
from unittest import TestCase
import graph_matching.utils.graph_processing as graph_processing
from graph_matching.utils.graph_processing import get_graph_from_pickle


class Test(TestCase):
    def test_get_graph_from_picle(self):
        g0 = get_graph_from_pickle("../graph_for_test/graph_00000.gpickle")
        self.assertTrue(len(g0.nodes) == 18)
        self.assertTrue(type(g0) == nx.Graph)
