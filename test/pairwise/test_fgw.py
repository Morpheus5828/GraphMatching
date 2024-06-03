"""This module contains for Fused Gomov Wasserstein algorithm
..moduleauthor:: Marius Thorre
"""

from unittest import TestCase
import numpy as np

import graph_matching.pairwise.fgw as fgw


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
distance_G1_G2 = fgw._M(G1_coord, G2_coord)

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
distance_G5_G6 = fgw._M(G5_coord, G6_coord)

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
distance_G7_G8 = fgw._M(G5_coord, G6_coord)



class Testfgw(TestCase):
    def test_get_M(self):
        M_G1_G2 = np.array([
            [10., 11., 11.04536102, 10.04987562],
            [9., 10., 10.04987562, 9.05538514],
            [10.04987562, 11.04536102, 11., 10.]
        ])
        self.assertTrue(np.allclose(fgw._M(G1_coord, G2_coord), M_G1_G2))
        M_G5_G6 = np.array([
            [8., 9.48683298, 11.04536102, 11.18033989],
            [5., 6.70820393, 8.06225775, 8.60232527],
            [5., 5., 7.28010989, 6.32455532],
            [8.60232527, 8.24621125, 10.77032961, 9.]
        ])

        self.assertTrue(np.allclose(fgw._M(G5_coord, G6_coord), M_G5_G6))

    def test_get_constant(self):
        transport_G1_G2 = mu_G1 @ mu_G2.T
        self.assertTrue(0 == fgw._get_constant(C1, C2, distance_G1_G2, transport_G1_G2))
        transport_G5_G6 = mu_G5 @ mu_G6.T
        self.assertTrue(0 == fgw._get_constant(C5, C6, distance_G5_G6, transport_G5_G6))

    def test_get_gradient(self):
        c_C1_C2 = fgw._get_constant(C1, C2, distance_G1_G2, mu_G1 @ mu_G2.T)
        transport_G1_G2 = mu_G1 @ np.transpose(mu_G2)
        result = np.array([
            [48., 58.5, 59., 48.5],
            [38.5, 48., 48.5, 39.],
            [48.5, 59., 58.5, 48.]
        ])
        self.assertTrue(np.allclose(fgw._get_gradient(c_C1_C2, C1, C2, distance_G1_G2, transport_G1_G2), result))

        c_C5_C6 = fgw._get_constant(C5, C6, distance_G5_G6, mu_G5 @ mu_G6.T)
        transport_G5_G6 = mu_G5 @ np.transpose(mu_G6)
        result = np.array([
            [20., 37., 53., 58.5],
            [0.5, 14.5, 24.5, 33.],
            [0.5, 4.5, 18.5, 16.],
            [25., 26., 50., 36.5]
        ])
        self.assertTrue(np.allclose(fgw._get_gradient(c_C5_C6, C5, C6, distance_G5_G6, transport_G5_G6), result))

    def test_solve_OT(self):
        transport_G1_G2 = mu_G1 @ mu_G2.T
        c_C1_C2 = fgw._get_constant(C1, C2, distance_G1_G2, transport_G1_G2)
        gradient_G1_G2 = fgw._get_gradient(c_C1_C2, C1, C2, distance_G1_G2, transport_G1_G2)
        result = np.array([
            [[0.34789425, 0.2275417,  0.17887392, 0.24569012],
             [0.20484411, 0.36419311, 0.28629763, 0.14466515],
            [0.20115067, 0.13156345, 0.28113554, 0.38615034]]
        ])

        self.assertTrue(
            np.allclose(
                fgw._solve_OT(
                    mu_s=mu_G1,
                    mu_t=mu_G2,
                    gradient=gradient_G1_G2,
                    gamma=1,
                    method="sinkhorn",
                    tolerance=1),
                result)
        )

        transport_G5_G6 = mu_G5 @ mu_G6.T
        c_C5_C6 = fgw._get_constant(C5, C6, distance_G5_G6, transport_G5_G6)
        gradient_G5_G6 = fgw._get_gradient(c_C5_C6, C5, C6, distance_G5_G6, transport_G5_G6)
        result = np.array([
            [9.99977963e-01, 1.75259061e-05, 4.51082836e-06, 4.34872387e-10],
            [9.64408851e-01, 3.39496019e-04, 3.52514835e-02, 1.69199659e-07],
            [3.60516168e-02, 2.79539422e-01, 5.31628002e-01, 1.52780959e-01],
            [2.57644170e-03, 4.01256502e-01, 3.46451183e-05, 5.96132411e-01]
        ])

        self.assertTrue(
            np.allclose(
                fgw._solve_OT(
                    mu_s=mu_G5,
                    mu_t=mu_G6,
                    gradient=gradient_G5_G6,
                    gamma=1,
                    method="sinkhorn",
                    tolerance=1
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
            tolerance=0.1,
            method="sinkhorn"
        )
        self.assertTrue(fgw._line_search(c_C1_C2, C1, C2, distance_G1_G2, transport_G1_G2, new_transport_G1_G2 == 1))

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
            tolerance=0.1
        )
        self.assertTrue(
            fgw._line_search(
                c_C5_C6,
                C5,
                C6,
                distance_G5_G6,
                transport_G5_G6,
                new_transport_G5_G6 == 1
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
            gamma=0.1,
            ot_method="sinkhorn",
            tolerance=0.1
        )
        match = np.zeros((C7.shape[0],)) - 1.0
        for i in range(transport.shape[0]):
            match[i] = np.argmax(transport[i, :])
        permut = np.array([3., 0., 3., 0.])
        print(match)
        self.assertTrue(np.linalg.norm(match - permut) < 1e-7)


