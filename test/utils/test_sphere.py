"""This module test sphere.py

..moduleauthor:: Marius Thorre
"""
import sys, os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import graph_matching.utils.graph.sphere as sphere

sphere_test = sphere.Sphere()


class SphereTest(TestCase):
    def test_make_sphere(self):
        tau = 1e-3  # tolerance
        x, y, z = sphere_test.make_sphere(radius=1.0)
        self.assertTrue(x.shape == (100, 100))
        self.assertTrue(y.shape == (100, 100))
        self.assertTrue(z.shape == (100, 100))
        self.assertTrue(np.linalg.norm(1 - np.max(x)) < tau)
        self.assertTrue(np.linalg.norm(-1 - np.min(x)) < tau)
        self.assertTrue(np.linalg.norm(1 - np.max(y)) < tau)
        self.assertTrue(np.linalg.norm(-1 - np.min(y)) < tau)
        self.assertTrue(np.linalg.norm(1 - np.max(z)) < tau)
        self.assertTrue(np.linalg.norm(-1 - np.min(z)) < tau)

    def test_draw_sphere(self):
        ax = plt.figure().add_subplot(projection='3d')
        sphere_test.draw_sphere(ax=ax, radius=1.0)
        # plt.show()

    def test_plot(self):
        sphere_test.plot(radius=1.0, data=sphere_test.make_sphere(radius=1.0))
        # plt.show()

    def test_sample(self):
        x, y, z = sphere_test.sample(
            nb_sample=2,
            radius=1.0,
            distribution="uniform",
        )
        self.assertTrue(x.shape == (2,))
        self.assertTrue(y.shape == (2,))
        self.assertTrue(z.shape == (2,))
