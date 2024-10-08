"""This module contains for Fused Gomov Wasserstein algorithm
..moduleauthor:: Marius Thorre
"""

from unittest import TestCase
import numpy as np

import graph_matching.algorithms.pairwise.fgw as fgw
from graph_matching.utils.graph_processing import _compute_distance
from graph_matching.algorithms.pairwise.pairwise_tools import _get_gradient, _get_constant

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
mu_G2 = mu_G2.reshape((-1, 1))
distance_G1_G2 = _compute_distance(G1_coord, G2_coord)

C5 = np.array([
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0]
])
C6 = np.array([
    [1, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])
G5_coord = np.array([
    [2, 6],
    [5, 6],
    [6, 3],
    [3, 1]
])
G6_coord = np.array([
    [10, 6],
    [11, 3],
    [13, 5],
    [12, 1]
])
mu_G5 = np.array([1, 1, 1, 1])
mu_G5 = mu_G5.reshape((-1, 1))
mu_G6 = np.array([1, 1, 1, 1])
mu_G6 = mu_G6.reshape((-1, 1))
distance_G5_G6 = _compute_distance(G5_coord, G6_coord)

C7 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
C8 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
G7_coord = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])
G8_coord = np.array([
    [10, 0],
    [11, 0],
    [11, 11],
    [10, 11]
])
mu_G7 = np.array([45, 26, 13, 12])
mu_G7 = mu_G7.reshape((-1, 1))
mu_G8 = np.array([45, 16, 18, 32])
mu_G8 = mu_G8.reshape((-1, 1))
distance_G7_G8 = _compute_distance(G5_coord, G6_coord)


class Testfgw(TestCase):
    def test_solve_OT(self):
        transport_G1_G2 = mu_G1 @ mu_G2.T
        c_C1_C2 = fgw._get_constant(C1, C2, distance_G1_G2, transport_G1_G2)
        gradient_G1_G2 = fgw._get_gradient(c_C1_C2, C1, C2, distance_G1_G2, transport_G1_G2)
        result = np.array([
            [[0.34635879, 0.23684603, 0.17994916, 0.23684603],
             [0.20182061, 0.37514562, 0.28502542, 0.13800836],
            [0.20182061 ,0.13800836, 0.28502542, 0.37514562]]
        ])

        self.assertTrue(
            np.allclose(
                fgw._solve_OT(
                    mu_s=mu_G1,
                    mu_t=mu_G2,
                    gradient=gradient_G1_G2,
                    gamma=1,
                    method="sinkhorn",
                    ),
                result)
        )

        transport_G5_G6 = mu_G5 @ mu_G6.T
        c_C5_C6 = fgw._get_constant(C5, C6, distance_G5_G6, transport_G5_G6)
        gradient_G5_G6 = fgw._get_gradient(c_C5_C6, C5, C6, distance_G5_G6, transport_G5_G6)
        result = np.array([
            [[9.42641964e-01, 5.55816621e-02 ,1.77467239e-03, 1.70138827e-06],
            [5.73381963e-02 ,6.79066386e-02 ,8.74713414e-01, 4.17510675e-05],
            [2.00681062e-05, 5.23503290e-01, 1.23507986e-01, 3.52968655e-01],
            [6.73733566e-07, 3.53008087e-01, 3.78107648e-06, 6.46987458e-01]]
        ])

        self.assertTrue(
            np.allclose(
                fgw._solve_OT(
                    mu_s=mu_G5,
                    mu_t=mu_G6,
                    gradient=gradient_G5_G6,
                    gamma=1,
                    method="sinkhorn",
                ),
                result)
        )

    def test_line_search(self):
        transport_G1_G2 = mu_G1 @ mu_G2.T
        c_C1_C2 = fgw._get_constant(
            C1=C1,
            C2=C2,
            distance=distance_G1_G2,
            transport=transport_G1_G2
        )
        gradient_G1_G2 = fgw._get_gradient(
            c_C1_C2=c_C1_C2,
            C1=C1,
            C2=C2,
            distance=distance_G1_G2,
            transport=transport_G1_G2
        )
        new_transport_G1_G2 = fgw._solve_OT(
            mu_s=mu_G1,
            mu_t=mu_G2,
            gradient=gradient_G1_G2,
            gamma=0.1,
            method="sinkhorn"
        )
        self.assertFalse(fgw._line_search(c_C1_C2, C1, C2, distance_G1_G2, transport_G1_G2, new_transport_G1_G2))

        transport_G5_G6 = mu_G5 @ mu_G6.T
        c_C5_C6 = fgw._get_constant(
            C1=C5,
            C2=C6,
            distance=distance_G5_G6,
            transport=transport_G5_G6
        )
        gradient_G5_G6 = fgw._get_gradient(
            c_C1_C2=c_C5_C6,
            C1=C5,
            C2=C6,
            distance=distance_G5_G6,
            transport=transport_G5_G6
        )
        new_transport_G5_G6 = fgw._solve_OT(
            mu_s=mu_G5,
            mu_t=mu_G6,
            gradient=gradient_G5_G6,
            gamma=0.1,
            method="sinkhorn",
        )
        self.assertTrue(
            fgw._line_search(
                c_C5_C6,
                C5,
                C6,
                distance_G5_G6,
                transport_G5_G6,
                new_transport_G5_G6
            )
        )

    def test_conditional_gradient(self):
        transport = fgw.conditional_gradient(
            mu_s=mu_G1,
            mu_t=mu_G2,
            C1=C1,
            C2=C2,
            distance=distance_G1_G2,
            gamma=0.01,
            ot_method="sinkhorn",
            tolerance=1e-6,
        )
        match = np.zeros((C1.shape[0],)) - 1.0
        for i in range(transport.shape[0]):
            match[i] = np.argmax(transport[i, :])
        permut = np.array([0, 2, 3])
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7)

        transport = fgw.conditional_gradient(
            mu_s=mu_G5,
            mu_t=mu_G6,
            C1=C5,
            C2=C6,
            distance=distance_G5_G6,
            gamma=0.1,
            ot_method="sinkhorn",
            tolerance=0.1
        )
        match = np.zeros((C5.shape[0],)) - 1.0
        for i in range(transport.shape[0]):
            match[i] = np.argmax(transport[i, :])
        permut = np.array([0., 2., 3., 3.])
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7)

        transport = fgw.conditional_gradient(
            mu_s=mu_G7,
            mu_t=mu_G8,
            C1=C7,
            C2=C8,
            distance=distance_G7_G8,
            eta=2,
            rho=20,
            N1=50,
            N2=50,
            ot_method="sns",
            tolerance=1e-6
        )
        match = np.zeros((C7.shape[0],)) - 1.0
        for i in range(transport.shape[0]):
            match[i] = np.argmax(transport[i, :])
        permut = np.array([0, 0., 0, 0.])
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7)


