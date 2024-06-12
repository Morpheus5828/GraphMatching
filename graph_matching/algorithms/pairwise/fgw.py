""" This module contains a other matching algoritm between pair of graph
Implementation of Vayer.T, Chapel.L, Flammary.R, Tavenard.R, Courty.N
    Optimal Transport for Structured data with application on graphs
    In International Conference on Machine Learning (pp. 6275-6284). PMLR.

With correction from Thesis of Cédric Vincent-Cuaz

.. moduleauthor:: Marius Thorre
"""

import numpy as np
from graph_matching.algorithms.solver import sinkhorn, sns, fx_sns
import math


def _get_gradient(c_C1_C2, C1, C2, distance, transport, alpha=0.5, q=2.0):
    """Compute Gradient using eq (7)
    :param np.ndarray c_C1_C2: constant from eq (6)
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param np.ndarray transport: vector which contains current transportation map
    :param float alpha: fixed to 1/2, it's the equilibrum between cost and distances
    :param float q: fixed to 2.0, it's L2 Loss parametter
    :return: gradient
    """
    return (1 - alpha) * (distance ** q) + 2.0 * alpha * c_C1_C2 - C1 @ transport @ (2.0 * C2).T


def _solve_OT(
        mu_s,
        mu_t,
        gradient,
        tolerance,
        gamma=None,
        rho=None,
        eta: float = None,
        method: str = "sinkhorn",
        N1=None,
        N2=None
) -> np.ndarray:
    """Solve Optimal Transport using Sinkhorn-Knopp method
    :param np.ndarray mu_s: starting probabilities of the sources nodes
    :param np.ndarray mu_t: starting probabilities of the target nodes
    :param float gamma: the strength of the regularization for the OT solver
    :param int iteration: number of iteration for Sinkhorn-Knopp
    :return: next transport matrix
    """
    if method == "sinkhorn":
        transport, _ = sinkhorn.sinkhorn_method(
            x=gradient,
            mu_s=np.squeeze(mu_s),
            mu_t=np.squeeze(mu_t),
            gamma=gamma,
            iterations=1000,
            tolerance=tolerance
        )
        return transport
    elif method == "sns":
        transport, _ = sns.sinkhorn_newton_sparse(
            cost=gradient,
            mu_s=mu_s,
            mu_t=mu_t,
            rho=rho,
            eta=eta,
            N1=N1,
            N2=N2,
            tolerance=tolerance
        )
        return transport
    elif method == "fx_sns":
        transport = fx_sns.sinkhorn_newton_sparse_method(
            cost=gradient,
            mu_s=mu_s,
            mu_t=mu_t,
            rho=rho,
            eta=eta,
            n1_iterations=N1,
            n2_iterations=N2,
            tolerance=tolerance
        )
        return transport
    else:
        print("Algo not recognized")


def _get_constant(
        C1: np.ndarray,
        C2: np.ndarray,
        distance: np.ndarray,
        transport: np.ndarray,
        alpha: float = 0.5
) -> np.ndarray:
    """Compute constant from eq (6) in Gromov-Wasserstein Averaging of Kernel and Distance Matrices
    by Peyré.G, Cuturi.M, Solomon.J
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param np.ndarray transport: vector which contains current transportation map
    :param alpha:
    :return: float
    """
    transport = transport.reshape((-1, 1))
    result = transport.flatten().T @ (-2 * alpha * np.kron(C2, C1)) @ transport.flatten()
    result += ((1 - alpha * distance).T).flatten() @ transport.flatten()
    return np.argmin(result)


def _M(adj_s, adj_t):
    """Compute euclidian distance between A and B adjacency matrix.
    :param np.ndarray A: adjacency matrix from source_graph
    :param np.ndarray B: adjacency matrix from target_graph
    :return: distance matrix
    """
    dist = np.zeros((adj_s.shape[0], adj_t.shape[0]))
    for a, i in zip(adj_s, range(len(adj_s))):
        for b, j in zip(adj_t, range(len(adj_t))):
            dist[i, j] = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return dist


def _line_search(c_C1_C2, C1, C2, distance, transport, new_transport, alpha=0.5):
    """Compute tau variable
    :param np.ndarray c_C1_C2: constant from eq (6)
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param np.ndarray transport: vector which contains current transportation map
    :param np.ndarray new_transport: vector which contains new transportation map
    :return: parameter tau
    """

    a = -2 * alpha * (
            (C1 @ (new_transport - transport)) @
            (C2 @ (new_transport - transport).T)
    ).sum()
    b = (((1 - alpha) * distance + alpha * c_C1_C2) * (new_transport - transport)).sum()
    b -= 2 * alpha * (
            ((C1 @ new_transport @ C2) * transport).sum() +
            ((C1 @ transport @ C2) * transport).sum()
    )

    tau = 0
    if a > 0:
        tau = np.min([1.0, np.max([0.0, -b / (2.0 * a)])])
    else:
        if a + b < 0:
            tau = 1
    return tau


def conditional_gradient(
        mu_s,
        mu_t,
        C1,
        C2,
        distance,
        gamma: float = None,
        rho: float = None,
        eta: float = None,
        N1: int = None,
        N2: int = None,
        tolerance: float = None,
        ot_method="sinkhorn",

):
    """Compute Condition Gradient for FGW
    :param np.ndarray mu_s: starting probabilities of the sources nodes
    :param np.ndarray mu_t: starting probabilities of the target nodes
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param float gamma: the strength of the regularization for the OT solver
    :return: transport map
    """
    mu_s = mu_s.reshape((-1, 1))
    mu_t = mu_t.reshape((-1, 1))

    transport = mu_s @ mu_t.T
    last_transport = []
    last_diff = 0
    n = mu_s.shape[0]
    for index in range(n):
        # 1 Gradient
        c_C1_C2 = _get_constant(C1=C1, C2=C2, distance=distance, transport=transport)
        gradient = _get_gradient(c_C1_C2=c_C1_C2, C1=C1, C2=C2, distance=distance, transport=transport)
        if np.all(gradient) != 0: gradient = gradient / np.max(gradient)
        # 2 OT
        new_transport = _solve_OT(
            mu_s=mu_s,
            mu_t=mu_t,
            gradient=gradient,
            gamma=gamma,
            rho=rho,
            eta=eta,
            method=ot_method,
            N1=N1,
            N2=N2,
            tolerance=tolerance
        )

        # 3 Line Search
        tau = _line_search(c_C1_C2, C1, C2, distance, transport, new_transport)
        # 4 Update
        transport = (1 - tau) * transport + tau * new_transport

        # 5 Check tolerance
        if index == 0:
            last_transport = transport
            last_diff = np.linalg.norm(transport - last_transport)
        elif index > 1:
            diff = np.linalg.norm(transport - last_transport)
            if np.linalg.norm(last_diff - diff) < tolerance:
                return transport

    return transport
