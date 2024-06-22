"""This module test display_graph_tools.py
..moduleauthor:: Marius Thorre
"""

import numpy as np
from unittest import TestCase
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.utils.graph_processing import get_graph_from_pickle

g0 = get_graph_from_pickle("../graph_for_test/graph_00000.gpickle")


class Test(TestCase):
    def test_transform(self):
        v = Visualisation(graph=g0)
        v.transform()
        truth = np.array([[29.43356677, -91.57760716, -27.33508759],
                 [50.07260194, 84.66921663, 17.99606317],
                 [72.92103209, -20.64689807, 65.23977835],
                 [25.16609113, 70.37393958, 66.43926915],
                 [14.99627927, 33.40922824, 93.05339906],
                 [-88.86556417, -44.87844211, 9.42533491],
                 [-26.37012968, -14.36650407, -95.38458901],
                 [95.50948644, -24.3670472, -16.85778782],
                 [-48.95906017, 10.20352116, 86.59618111],
                 [-9.00334241, 44.70445505, -88.99691862],
                 [49.30969484, -41.57616795, 76.41973733],
                 [3.90626271, 66.26822449, -74.78812429],
                 [47.15251309, 78.83678445, -39.51457867],
                 [-72.90243545, -57.12819622, 37.70416558],
                 [-68.38127449, 64.71095903, 33.71191303],
                 [61.39557924, -60.05583211, 51.22382139],
                 [-28.73941053, -95.54038341, 6.7883297],
                 [60.85410147, 70.22896859, 36.94144428]])
        self.assertTrue(np.allclose(v.points, truth))

    def test_check_point_on_sphere(self):
        v = Visualisation(graph=g0)
        self.assertTrue(v.check_point_on_sphere(v.points, 100))
        self.assertFalse(v.check_point_on_sphere(v.points, 10))

    def test_construct_sphere(self):
        v = Visualisation(graph=g0)
        v.construct_sphere()
        self.assertTrue(v.fig is not None)


