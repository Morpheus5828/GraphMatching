"""This module test FUGW implemntation

.. moduleauthor:: Marius Thorre
"""

from unittest import TestCase
import numpy as np
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
#mu_G2 = mu_G2.reshape((-1, 1))
distance_G1_G2 = fgw._M(G1_coord, G2_coord)


class TestFugw(TestCase):
    def test_geometry_cost(self):
        n = 2
        p = 3
        D_s = np.random.randint(0, 1, size=(n, n))
        D_t = np.random.randint(0, 1, size=(p, p))
        self.assertEquals(fugw._geometry_cost(D_s, D_t, n, p).shape, (2, 3, 2, 3))

    def test_kron(self):
        G = np.random.randint(0, 1, size=(2, 3, 2, 3))
        P = np.random.randint(0, 1, size=(2, 3))

        self.assertEquals(fugw._kron(G, P).shape, (2, 3))

    def test_cost(self):
        P = np.kron(mu_G1, mu_G2) / np.sqrt(np.sum(mu_G1) * np.sum(mu_G2))

        fugw._cost(
            P=P,
            G=fugw._geometry_cost(C1, C2, C1.shape[0], C2.shape[0]),
            C=np.array([
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0]
            ]),
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            alpha=0.1,
            epsilon=1
        )

    def test_scaling(self):
        P = np.kron(mu_G1, mu_G2) / np.sqrt(np.sum(mu_G1) * np.sum(mu_G2))

        c_p = fugw._cost(
            P=P,
            G=fugw._geometry_cost(C1, C2, C1.shape[0], C2.shape[0]),
            C=np.array([
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0]
            ]),
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            alpha=1,
            epsilon=1
        )

        fugw._scaling(
            C=c_p,
            w_s=mu_G1,
            w_t=mu_G2,
            rho=1,
            epsilon=1
        )

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

        print(P, Q, i)
