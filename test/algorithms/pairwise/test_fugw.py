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

        self.assertEquals(fugw._geometry_cost(D_s, D_t).shape, (2, 3, 2, 3))

    def test_kron(self):
        G = np.random.randint(0, 1, size=(2, 3, 2, 3))
        P = np.random.randint(0, 1, size=(2, 3))
        self.assertEquals(fugw._kron_tensor(G, P).shape, (2, 3))

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
        self.assertTrue(np.allclose(result, truth))

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

    def test_LB_FUGW_graph(self):
        g_src_nodes = []
        g_src_adj = nx.adjacency_matrix(G0).todense()
        for index in range(len(G0.nodes)):
            if len(G0.nodes[index]) > 0:
                g_src_nodes.append(G0.nodes[index]["coord"])
        g_src_nodes = np.array(g_src_nodes)

        g_target_nodes = []
        g_target_adj = nx.adjacency_matrix(G10).todense()
        for index in range(len(G10.nodes)):
            if len(G10.nodes[index]) > 0:
                g_target_nodes.append(G10.nodes[index]["coord"])
        g_target_nodes = np.array(g_target_nodes)
        g_target_nodes = g_target_nodes.reshape(g_target_nodes.shape[1], g_target_nodes.shape[0])

        g_src_nodes /= 100
        g_target_nodes /= 100

        distance = []
        for i in g_src_nodes:
            for j in g_target_nodes.T:
                tmp = 0
                tmp += np.abs(i[0] - j[0])
                tmp += np.abs(i[1] - j[1])
                tmp += np.abs(i[2] - j[2])
                distance.append(tmp)

        distance = np.array(distance).reshape(30, 30)

        w_s = np.ones(shape=(30, 1)) / 30
        w_t = np.ones(shape=(1, 30)) / 30

        c_src_dest = _get_constant(
            C1=g_src_adj,
            C2=g_target_adj,
            distance=distance,
            transport=w_s @ w_t
        )
        cost = _get_gradient(
            c_C1_C2=c_src_dest,
            C1=g_src_adj,
            C2=g_target_adj,
            distance=distance,
            transport=w_s @ w_t
        )

        distance = fugw._geometry_cost(g_src_adj, g_target_adj)

        # rho = 0.1
        # alpha = 0.25
        # epsilon = 1e-05
        #
        # P, Q = fugw.LB_FUGW(
        #     cost=cost,
        #     distance=distance,
        #     w_s=w_s,
        #     w_t=w_t,
        #     rho=rho,
        #     alpha=alpha,
        #     epsilon=epsilon
        # )
        #
        # print(P)

        rho_values = [0.1, 1, 10, 100]
        alpha_values = [0.25, 0.5, 0.75]
        epsilon_values = [1e-02, 1e-03, 1e-04, 1e-05]

        for rho in rho_values:
            for alpha in alpha_values:
                for epsilon in epsilon_values:
                    try:
                        P, Q = fugw.LB_FUGW(
                            cost=cost,
                            distance=distance,
                            w_s=w_s,
                            w_t=w_t,
                            rho=rho,
                            alpha=alpha,
                            epsilon=epsilon
                        )

                        if not np.isnan(P).any() and not np.isnan(Q).any():
                            print(f"Valid parameters found: rho={rho}, alpha={alpha}, epsilon={epsilon}")

                    except Exception as e:
                        print(f"Exception for parameters rho={rho}, alpha={alpha}, epsilon={epsilon}: {e}")
