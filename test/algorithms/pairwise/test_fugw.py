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
from graph_matching.utils.graph_processing import get_graph_from_pickle
from graph_matching.utils.graph_processing import get_graph_coord
import graph_matching.algorithms.pairwise.fugw as fugw
import graph_matching.algorithms.pairwise.fgw as fgw

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
mu_G1 = np.array([1, 1, 1])
mu_G1 = mu_G1.reshape((-1, 1))
mu_G2 = np.array([1, 1, 1, 1])
mu_G2 = mu_G2.reshape((1, -1))
distance_G1_G2 = fgw._M(G1_coord, G2_coord)

G0 = get_graph_from_pickle(
    os.path.join(
        project_root,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_1_outliers_varied",
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
        "noise_1_outliers_varied",
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
            [-2.57193376, -2.12193376, -2.12193376, -2.12193376],
            [-2.57193376, -2.57193376, -2.12193376, -2.57193376],
            [-2.57193376, -2.57193376, -2.12193376, -2.57193376]])
        self.assertTrue(np.allclose(result, truth))

    def test_scaling(self):
        P = np.kron(mu_G1, mu_G2) / np.sqrt(np.sum(mu_G1) * np.sum(mu_G2))
        cost = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ])
        c_p = fugw._cost(
            P=P,
            G=fugw._geometry_cost(C1, C2),
            C=cost,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            alpha=1,
            epsilon=1
        )
        result = fugw._scaling(
            C=c_p,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            epsilon=1
        )

        # truth = np.array([
        #     [2.22328771e-06, 2.22328771e-06, 2.22328771e-06, 2.22328771e-06],
        #     [2.22328771e-06, 2.22328771e-06, 2.22328771e-06, 2.22328771e-06],
        #     [2.22328771e-06, 2.22328771e-06, 2.22328771e-06, 2.22328771e-06]
        # ])
        #
        # self.assertTrue(np.allclose(result, truth))

    def test_LB_FUGW(self):
        cost = G1_coord @ G2_coord.reshape(G2_coord.shape[1], G2_coord.shape[0])
        P, Q, i = fugw.LB_FUGW(
            cost=cost,
            distance=fugw._geometry_cost(C1, C2),
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            alpha=0.5,
            epsilon=108,
            return_i=True
        )
        truth_p = np.array([
            [0.95276443, 0.95276443, 0.95276443, 0.95276443],
            [0.95276443, 0.95276443, 0.95276443, 0.95276443],
            [0.95276443, 0.95276443, 0.95276443, 0.95276443]
        ])
        self.assertTrue(np.allclose(truth_p, P))
        truth_q = np.array([
            [0.9544022,  0.95652338, 0.95459484, 0.95652338],
            [0.95247794, 0.95633035, 0.95247794, 0.95633035],
            [0.95652338, 0.95652338, 0.95652338, 0.95652338]
        ])
        self.assertTrue(np.allclose(truth_q, Q))

        self.assertEquals(i, 8)

    def test_LB_FUGW_graph(self):
        G_source = G0
        G_dest = G10

        mu_s = np.ones(nx.number_of_nodes(G_source)) / nx.number_of_nodes(G_source)
        mu_s = mu_s.reshape((-1, 1))
        mu_t = np.ones(nx.number_of_nodes(G_dest)) / nx.number_of_nodes(G_dest)

        adj_matrix_s = nx.adjacency_matrix(G_source).toarray()
        adj_matrix_t = nx.adjacency_matrix(G_dest).toarray()

        graph_coord_s = get_graph_coord(G_source, nb_dimension=3)
        graph_coord_t = get_graph_coord(G_dest, nb_dimension=3)

        cost = graph_coord_s @ graph_coord_t.reshape(graph_coord_t.shape[1], graph_coord_t.shape[0])
        P, Q = fugw.LB_FUGW(
            cost=cost,
            distance=fugw._geometry_cost(adj_matrix_s, adj_matrix_t),
            w_s=mu_s,
            w_t=mu_t,
            rho=1,
            epsilon=500,
            alpha=0.5
        )

        print(P, Q)




