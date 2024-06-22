"""Concatenation.py file Test
..moduleauthor:: Marius THORRE
"""

from unittest import TestCase
import numpy as np
import graph_matching.utils.concatenation as concatenation


class TestManOpt(TestCase):
    def test_get_top_left(self):
        result = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        self.assertTrue(np.allclose(concatenation.get_top_left(np.ones((3, 2))), result))

    def test_get_bottom_right(self):
        result = np.array([
            [3, 0],
            [0, 3]
        ])
        self.assertTrue(np.allclose(concatenation.get_bottom_right(np.ones((3, 2))), result))

    def test_fusion(self):
        result = np.array([
            [2, 0, 0, 1, 1],
            [0, 2, 0, 1, 1],
            [0, 0, 2, 1, 1],
            [1, 1, 1, 3, 0],
            [1, 1, 1, 0, 3],
        ])
        self.assertTrue(np.allclose(concatenation.fusion(np.ones((3, 2))), result))



