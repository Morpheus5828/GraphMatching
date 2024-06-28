"""This module test display_graph_tools.py
..moduleauthor:: Marius Thorre
"""
import os, sys
import numpy as np
from unittest import TestCase
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.utils.graph_processing import get_graph_from_pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

g0_path = "resources/graph_for_test/generation/noise_10_outliers_varied/graph_00000.gpickle"
g0 = get_graph_from_pickle(os.path.join(project_path, g0_path))


class Test(TestCase):
    def test_transform(self):
        v = Visualisation(graph=g0)
        v.extract_coord_label()
        truth = np.array([
            [-55.49753476, -82.15395423, -13.06718944],
            [1.88299524,  36.15755778,  93.21526347],
            [27.34182859,  92.55916943, -26.17679437],
            [-24.71182553,   9.41527848, -96.4400239],
            [-4.85208243,  70.55715679,  70.69755952],
            [-14.33961289,  65.62248738,-74.08147307]
        ])
        self.assertTrue(np.allclose(v.points, truth))

    def test_construct_sphere(self):
        v = Visualisation(graph=g0)
        v.extract_coord_label()
        v.construct_sphere()
        self.assertTrue(v.fig is not None)
