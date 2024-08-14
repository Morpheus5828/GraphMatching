"""This module test FUGW implemntation

.. moduleauthor:: Marius Thorre
"""

import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from unittest import TestCase
import numpy as np
import networkx as nx
from graph_matching.utils.graph_processing import get_graph_from_pickle, _compute_distance
from graph_matching.algorithms.pairwise.pairwise_tools import _get_gradient, _get_constant
import graph_matching.algorithms.pairwise.fugw as fugw


C1 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])
C2 = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])
G1_coord = np.array([
    [0, 1],
    [1, 1],
    [0, 0]
])
G2_coord = np.array([
    [10, 1],
    [11, 1],
    [11, 0],
    [10, 0]
])
mu_G1 = np.array([1, 1, 1]) / 3
mu_G1 = mu_G1.reshape((-1, 1))
mu_G2 = np.array([1, 1, 1, 1]) / 4
mu_G2 = mu_G2.reshape((1, -1))
distance_G1_G2 = _compute_distance(G1_coord, G2_coord)

G0 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_60",
        "graph_00000.gpickle"
    )
)

G10 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_60",
        "graph_00010.gpickle"
    )
)


class TestFugw(TestCase):
    def test_geometry_cost(self):
        n = 2
        p = 3
        D_s = np.random.randint(0, 1, size=(n, n))
        D_t = np.random.randint(0, 1, size=(p, p))

        self.assertEqual(fugw._geometry_cost(D_s, D_t).shape, (2, 3, 2, 3))

    def test_kron(self):
        G = np.random.randint(0, 1, size=(2, 3, 2, 3))
        P = np.random.randint(0, 1, size=(2, 3))
        self.assertEqual(fugw._kron_tensor(G, P).shape, (2, 3))

    def test_cost(self):
        P = np.kron(mu_G1, mu_G2) / np.sqrt(np.sum(mu_G1) * np.sum(mu_G2))
        cost = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ])
        result = fugw._cost(
            P=P,
            G=fugw._geometry_cost(C1, C2),
            C=cost,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            alpha=0.1,
            epsilon=1
        )

        truth = np.array([
            [-3.16308707, -2.71308707, -2.71308707, -2.71308707],
            [-3.16308707, -3.16308707, -2.71308707, -3.16308707],
            [-3.16308707, -3.16308707, -2.71308707, -3.16308707]]
        )
        print(result)


    def test_scaling(self):
        P = np.kron(mu_G1, mu_G2) / np.sqrt(np.sum(mu_G1) * np.sum(mu_G2))
        c_C1_C2 = _get_constant(
            C1=C1,
            C2=C2,
            distance=distance_G1_G2,
            transport=mu_G1 @ mu_G2
        )
        cost = _get_gradient(
            c_C1_C2=c_C1_C2,
            C1=C1, C2=C2,
            distance=distance_G1_G2,
            transport=mu_G1 @ mu_G2
        )

        rho = 1e-1
        alpha = 0.7
        epsilon = 1e-1
        c_p = fugw._cost(
            P=P,
            G=fugw._geometry_cost(C1, C2),
            C=cost,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=rho,
            alpha=alpha,
            epsilon=epsilon
        )
        result = fugw._scaling(
            C=c_p,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=rho,
            epsilon=epsilon
        )
        print(result)

        # truth = np.array([
        #     [9.52837905e-19, 2.51041408e-01, 5.77925403e-19, 2.51041408e-01],
        #     [1.38437751e-01, 7.03488133e-06, 8.39667404e-02, 7.03488133e-06],
        #     [1.38437751e-01, 7.03488133e-06, 8.39667404e-02, 7.03488133e-06]]
        # )
        #
        # self.assertTrue(np.allclose(result, truth))

    def test_LB_FUGW(self):
        c_C1_C2 = _get_constant(
            C1=C1,
            C2=C2,
            distance=distance_G1_G2,
            transport=mu_G1 @ mu_G2
        )
        cost = _get_gradient(
            c_C1_C2=c_C1_C2,
            C1=C1, C2=C2,
            distance=distance_G1_G2,
            transport=mu_G1 @ mu_G2
        )

        rho = 50
        alpha = 0.9
        epsilon = 1

        P, Q, i = fugw.LB_FUGW(
            cost=cost,
            distance=fugw._geometry_cost(C1, C2),
            w_s=mu_G1,
            w_t=mu_G2,
            rho=rho,
            alpha=alpha,
            epsilon=epsilon,
            return_i=True
        )

        print(P)

