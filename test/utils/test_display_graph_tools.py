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

g0_path = "resources/graph_for_test/generation/graph_00000.gpickle"
g0 = get_graph_from_pickle(os.path.join(project_path, g0_path))


class Test(TestCase):
    def test_transform(self):
        v = Visualisation(graph=g0)
        v.transform()
        truth = np.array([[63.26873894, -63.23596257, 44.702122],
                          [-85.2485662, -45.18006945, 26.2953092],
                          [99.79208327, 5.7780376, 2.85559076],
                          [5.62108774, -44.11735131, -89.56596835],
                          [-39.75616037, 67.3853749, 62.27888055],
                          [27.1514147, 89.00063922, -36.62904446],
                          [-95.31484109, -30.18146411, -2.03967917],
                          [63.26408812, 44.87498521, -63.11807076],
                          [94.02506592, -10.97143314, -32.2321987],
                          [-51.17179641, -11.09872108, 85.19545553],
                          [-96.10160237, 21.04233042, -17.9360629],
                          [17.3499715, 46.16718143, 86.99178034],
                          [18.71910857, -82.94558549, -52.62722511],
                          [-43.25941584, 12.99669466, 89.21720052],
                          [0.491645, -86.28260817, 50.54769839],
                          [-24.36399856, -67.80637477, 69.34472665],
                          [66.01544767, 56.4767599, 49.52106886],
                          [-38.95155784, -60.87401583, -69.11678767],
                          [-94.44573656, 22.96407866, 23.50859285],
                          [-78.58286659, 55.22118906, 27.84516757],
                          [-91.30953544, -17.90205132, -36.63448233],
                          [-79.67589754, 20.56718767, -56.82202163],
                          [-53.54028975, -39.8602028, -74.46208167],
                          [7.30733951, -98.52947617, -15.44490577],
                          [76.65521458, -64.1919302, 1.83689277],
                          [9.40908491, 46.92860185, -87.80191029],
                          [42.33243242, -11.86675398, -89.81728851],
                          [-25.17400719, 94.73340772, 19.79522223],
                          [49.36866735, 14.06560881, 85.81895672],
                          [-60.86780809, -65.31287599, 45.04817607]])
        self.assertTrue(np.allclose(v.points, truth))

    def test_check_point_on_sphere(self):
        v = Visualisation(graph=g0)
        v.transform()
        self.assertTrue(v.check_point_on_sphere(v.points, 100))
        self.assertFalse(v.check_point_on_sphere(v.points, 10))

    def test_construct_sphere(self):
        v = Visualisation(graph=g0)
        v.transform()
        v.construct_sphere()
        self.assertTrue(v.fig is not None)
