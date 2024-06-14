"""This module contains an optimal transport algorithm.

Implementation of Sinkhorn Newton Sparse from the paper
Accelerating sinkhorn algorithm with sparse newton iterations by
X.Tang, M.Shavlovsky, H.Rahmanian, E.Tardini, K.Koshy, T.Xiao, L.Ying


.. moduleauthor:: Marius THORRE
"""

import numpy as np
import graph_matching.utils.fgw.concatenation as concatenation
from scipy.sparse.linalg import cg


def _sinkhorn_stage(
        cost: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        N1: int,
        eta: float,
        mu_s: np.ndarray,
        mu_t: np.ndarray,
        tolerance: float
):
    """
    Sinkhorn stage part
    :param cost: nost matrix
    :param x: 
    :param y: 
    :param N1: 
    :param eta: 
    :param mu_s: 
    :param mu_t: 
    :param tolerance: 
    :return: 
    """
    i = 0
    P = 0
    previous_diff = 0
    previous_z = np.concatenate((x, y))
    while i < N1:
        P = np.exp(eta * (-cost +
                          (x @ np.ones((cost.shape[1], 1)).T) +
                          (np.ones((cost.shape[0], 1)) @ y.T)
                          ) - 1
                   )

        x_next = x + ((np.log(mu_s) - np.log(P @ np.ones((cost.shape[1], 1)))) / eta)

        P = np.exp(eta * (-cost +
                          (x @ np.ones((cost.shape[1], 1)).T) +
                          (np.ones((cost.shape[0], 1)) @ y.T)
                          ) - 1
                   )

        y_next = y + ((np.log(mu_t) - np.log(P.T @ np.ones((P.shape[0], 1)))) / eta)

        if i == 0:
            previous_diff = np.linalg.norm(previous_z - np.concatenate((x_next, y_next)))
        else:
            diff = np.linalg.norm(previous_z - np.concatenate((x_next, y_next)))
            if diff - previous_diff < tolerance:
                break
            previous_diff = diff

        previous_z = np.concatenate((x_next, y_next))
        x = x_next
        y = y_next
        i += 1

    return x, y, P, i


def _newton_stage(
        cost: np.ndarray,
        mu_s: np.ndarray,
        mu_t: np.ndarray,
        rho: float,
        eta: float,
        iteration: int,
        x: np.ndarray,
        y: np.ndarray,
        P: np.ndarray,
        N1: int,
        N2: int,
        tolerance: float
):
    i_init = iteration
    z = np.concatenate((x, y))
    previous_diff = 0
    while iteration < N1 + N2:
        delta_second = eta * concatenation.fusion(P)
        M = _sparsify(delta_second, rho)
        if np.all(M == 0):
            return np.exp(
                eta * (
                        -cost +
                        z[:x.shape[0]] @ np.ones((cost.shape[1], 1)).T +
                        np.ones((cost.shape[0], 1)) @ z[-y.shape[0]:].T
                ) - 1
            ), iteration

        gradient_z_x = mu_s - (P @ np.ones((P.shape[1], 1)))
        gradient_z_y = mu_t - (P.T @ np.ones((P.shape[0], 1)))

        gradient_z = np.zeros((gradient_z_x.shape[0] + gradient_z_y.shape[0], 1))

        gradient_z[:gradient_z_x.shape[0], :1] = gradient_z_x
        gradient_z[(gradient_z.shape[0] - gradient_z_y.shape[0]):] = gradient_z_y

        delta_z = _conjugate_gradient(M.T, gradient_z)
        alpha = _line_search(x, y, z, delta_z, eta, mu_s, mu_t)

        z_next = z + alpha * delta_z

        diff = np.linalg.norm(z - z_next)
        if i_init != iteration:
            if np.linalg.norm(diff - previous_diff) < tolerance:
                break

        previous_diff = diff
        iteration += 1

    return np.exp(
        eta * (
                -cost +
                z[:x.shape[0]] @ np.ones((cost.shape[1], 1)).T +
                np.ones((cost.shape[0], 1)) @ z[-y.shape[0]:].T
        ) - 1
    ), iteration


def _line_search(x, y, z, delta_z, eta, c, r):  # from eq (2)
    f_z = -1 / eta * (np.sum(np.exp(eta * (-c + x + y.T) - 1)))
    f_z += np.sum(c @ x.T) + np.sum(r @ y.T)

    alpha = 0.1

    f_alpha_delta_z = z + alpha * delta_z
    x_alpha_delta_z = f_alpha_delta_z[:x.shape[0]]
    y_alpha_delta_z = f_alpha_delta_z[-y.shape[0]:]

    f_alpha_delta_z = -1 / eta * (np.sum(np.exp(eta * (-c + x_alpha_delta_z + y_alpha_delta_z.T) - 1)))

    f_alpha_delta_z += np.sum(c @ x_alpha_delta_z.T) + np.sum(r @ y_alpha_delta_z.T)

    while f_z > f_alpha_delta_z:
        alpha /= 10
        f_alpha_delta_z = z + alpha * delta_z
        x_alpha_delta_z = f_alpha_delta_z[:x.shape[0]]
        y_alpha_delta_z = f_alpha_delta_z[-y.shape[0]:]

        f_alpha_delta_z = -1 / eta * (np.sum(np.exp(eta * (-c + x_alpha_delta_z + y_alpha_delta_z.T) - 1)))
        f_alpha_delta_z += np.sum(c @ x_alpha_delta_z.T) + np.sum(r @ y_alpha_delta_z.T)

    return alpha


def _sparsify(matrix, threshold):
    shape = matrix.shape
    matrix = matrix.reshape((-1, 1))
    matrix[::-1].sort()
    k = int((threshold * matrix.shape[0]) / 100)
    matrix[np.abs(matrix) < k] = 0
    return matrix.reshape(shape)


def _conjugate_gradient(A, b):
    x, exit_code = cg(A, b, atol=1e-3, maxiter=100)

    return x.reshape((-1, 1))


def sinkhorn_newton_sparse(
        cost: np.ndarray,
        mu_s: np.ndarray,
        mu_t: np.ndarray,
        rho: float,
        eta: float,
        N1: int,
        N2: int,
) -> (np.array, int):
    """ Sinkhorn Newton Stage algorithms
    :param cost: cost matrix
    :param mu_s:
    :param mu_t:
    :param rho:
    :param eta: regular parameter
    :param N1:
    :param N2:
    :return:
    """
    mu_s = mu_s.reshape((-1, 1))
    mu_t = mu_t.reshape((-1, 1))
    x = np.zeros((cost.shape[0], 1))
    y = np.zeros((cost.shape[1], 1))
    x, y, P, i = _sinkhorn_stage(cost, x, y, N1, eta, mu_s, mu_t, tolerance=1e-4)
    result, i = _newton_stage(cost, mu_s, mu_t, rho, eta, i, x, y, P, N1, N2, tolerance=1e-4)
    return result, i
