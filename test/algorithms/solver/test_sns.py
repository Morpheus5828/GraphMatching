import numpy as np
import graph_matching.algorithms.solver.sns as sns
from unittest import TestCase

cost = np.array([
    [.48, .585, .59, .485],
    [.385, .48, .485, .39],
    [.485, .59, .585, .48]
])

mu_s = np.array([1, 1, 1])
mu_s = mu_s.reshape((-1, 1))
mu_t = np.array([1, 1, 1, 1])
mu_t = mu_t.reshape((-1, 1))


class TestSns(TestCase):
    def test_sinkhorn_stage(self):
        x_init = np.zeros((cost.shape[0], 1))
        y_init = np.zeros((cost.shape[1], 1))

        x_result = np.array([
            [0.33909537],
            [0.23959371],
            [0.33909537]
        ])
        y_result = np.array([
            [0.39853407],
            [0.49973079],
            [0.50163832],
            [0.40041806]
        ])
        x_pred, y_pred, _, i = sns._sinkhorn_stage(
            cost=cost,
            x=x_init,
            y=y_init,
            N1=10,
            eta=2,
            mu_s=mu_s,
            mu_t=mu_t,
            tolerance=0.01
        )
        self.assertTrue(np.allclose(x_pred, x_result))
        self.assertTrue(np.allclose(y_pred, y_result))

    def test_newton_stage(self):
        x_init = np.zeros((cost.shape[0], 1))
        y_init = np.zeros((cost.shape[1], 1))

        x, y, P, iteration = sns._sinkhorn_stage(
            cost=cost,
            x=x_init,
            y=y_init,
            N1=10,
            eta=2,
            mu_s=mu_s,
            mu_t=mu_t,
            tolerance=0.01
        )

        result = np.array([
            [0.61585661, 0.61118983, 0.60742133, 0.61203052],
            [0.61033674, 0.61794797, 0.6141378, 0.60654494],
            [0.60972873, 0.60510839, 0.61352601, 0.61818153]
        ])
        transport, i = sns._newton_stage(
            cost=cost,
            mu_s=mu_s,
            mu_t=mu_t,
            rho=15,
            eta=2,
            iteration=iteration,
            x=x,
            y=y,
            P=P,
            N1=10,
            N2=10,
            tolerance=0.01
        )
        self.assertTrue(np.allclose(result, transport))

    def test_sinkhorn_newton_stage(self):
        transport, i = sns.sinkhorn_newton_sparse(
            cost=cost,
            mu_s=mu_s,
            mu_t=mu_t,
            rho=15,
            eta=1,
            N1=10,
            N2=20,
            tolerance=0.01
        )

        result = np.array([
            [0.37442765, 0.37309524, 0.37189778, 0.37322163],
            [0.37265326, 0.37505907, 0.3738553, 0.37145295],
            [0.37256018, 0.37123442, 0.37376192, 0.37509241]
        ])
        self.assertTrue(np.allclose(transport, result))
        self.assertTrue(i == 1)

    def test_conjugate_gradient(self):
        A = np.array([[4, 1], [1, 3]])
        b = np.array([[1], [2]])
        result = np.array([
            [0.09090909],
            [0.63636364]
        ])
        self.assertTrue(np.allclose(sns._conjugate_gradient(A, b), result))

    def test_sparsify(self):
        a = np.arange(64)
        a = a.reshape((8, 8))
        result = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 10, 11, 12, 13, 14, 15],
            [32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55],
            [56, 57, 58, 59, 60, 61, 62, 63]
        ])

        self.assertTrue(result.all() == sns._sparsify(a, 15).all())

