from unittest import TestCase
import numpy as np
import sys, os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
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

        p = np.array([
             [5.70840367e-08, 5.51822917e-08, 5.34031741e-08, 5.51822917e-08],
             [5.42774045e-08, 5.79873919e-08, 5.61178358e-08, 5.24691620e-08],
             [5.42774045e-08, 5.24691620e-08, 5.61178358e-08, 5.79873919e-08]
        ])

        print(concatenation.fusion(p))


