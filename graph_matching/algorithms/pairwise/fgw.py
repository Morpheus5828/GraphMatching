""" This module contains a other matching algoritm between a pair of graph
Implementation of Vayer.T, Chapel.L, Flammary.R, Tavenard.R, Courty.N
    Optimal Transport for Structured data with application on graphs
    In International Conference on Machine Learning (pp. 6275-6284). PMLR.

With correction from Thesis of Cédric Vincent-Cuaz

.. moduleauthor:: Marius Thorre
"""

import numpy as np
from graph_matching.algorithms.solver import sinkhorn, sns, fx_sns
from graph_matching.algorithms.pairwise.pairwise_tools import _get_gradient, _get_constant


def _solve_OT(
        mu_s,
        mu_t,
        gradient,
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
            iterations=1000
        )
        return transport
    elif method == "sns":
        transport, _ = sns.sinkhorn_newton_sparse(
            cost=gradient/np.max(gradient),
            mu_s=mu_s,
            mu_t=mu_t,
            rho=rho,
            eta=eta,
            N1=N1,
            N2=N2,
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
        )
        return transport
    else:
        print("Algo not recognized")


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
        tolerance: float = 1e-4,
        ot_method="sinkhorn",

):
    """Compute Condition Gradient for FGW
    :param np.ndarray mu_s: starting probabilities of the sources nodes
    :param np.ndarray mu_t: starting probabilities of the target nodes
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param float gamma: the strength of the regularization for the OT solver
    :param rho: Sinkhorn Newton Stage sparsify parametter
    :param eta: Sinkhorn Newton Stage regularization parametter
    :param N1: Sinkhorn part iteration in Sinkhron Newton Stage algorithm
    :param N2: Newton part iteration in Sinkhron Newton Stage algorithm
    :param tolerance: convergence parameter
    :param ot_method: optimal transport method used, sinkhorn or sns
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
        # 2 OT
        new_transport = _solve_OT(
            mu_s=mu_s,
            mu_t=mu_t,
            gradient=gradient/np.max(gradient),
            gamma=gamma,
            rho=rho,
            eta=eta,
            method=ot_method,
            N1=N1,
            N2=N2,
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
