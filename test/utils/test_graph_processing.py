"""This module will contain test from graph_processing.py file
..moduleauthor:: Marius Thorre
"""

import os, sys
import networkx as nx
from unittest import TestCase

from graph_matching.utils.graph_processing import get_graph_from_pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)



class Test(TestCase):
    def test_get_graph_from_picle(self):
        g0_path = "test/graph_for_test/graph_00000.gpickle"
        g0 = get_graph_from_pickle(os.path.join(project_path, g0_path))

        self.assertTrue(len(g0.nodes) == 30)
        self.assertTrue(type(g0) == nx.Graph)
