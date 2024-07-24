"""This module contains a matching method between a pair of graph
Implementation of Thual, A., Tran, Q. H., Zemskova, T., Courty, N., Flamary, R. et al.
Aligning individual brains with fused unbalanced Gromov Wasserstein
Advances in neural information processing systems, 35, 21792-21804

.moduleauthor:: Marius Thorre
"""

import numpy as np


def _geometry_cost(D_s: np.ndarray, D_t: np.ndarray) -> np.ndarray:
    n = D_s.shape[0]
    p = D_t.shape[0]

    result = []
    for i in range(n):
        for j in range(p):
            for k in range(n):
                for l in range(p):
                    result.append(np.abs(D_s[i, k] - D_t[j, l]))

    return np.array(result).reshape(n, p, n, p)


def _kron_tensor(G: np.ndarray, P: np.ndarray) -> np.ndarray:
    K = G.shape[2]
    L = G.shape[3]

    result = []
    for k in range(K):
        for l in range(L):
            result.append(np.sum(G[:, :, k, l] * P[:, :]))
    return np.array(result).reshape(K, L)


def _cost(
        P: np.ndarray,
        G: np.ndarray,
        C: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        alpha: float,
        epsilon: float
) -> np.ndarray:
    """
    Compute cost matrix
    :param P: Transport matrix, size (n,p)
    :param G: Geometry cost matrix, size (n*n, p*p)
    :param C: Feature cost matrix, size (n,p)
    :param w_s: source distribution vector, size (n, )
    :param w_t: target distribution vector, size (p, )
    :param rho: float hyperparameter in R+
    :param alpha: float hyperparameter in [0, 1]
    :param epsilon: float hyperparameter in R+
    :return: a cost matrix
    """
    c = alpha * _kron_tensor(G, P)
    c += ((1 - alpha) / 2) * C
    c += rho * (np.log(P.sum(axis=1) / w_s) * P.sum(axis=1)).sum()
    c += rho * (np.log(P.sum(axis=0) / w_t) * P.sum(axis=0)).sum()
    c += epsilon * (np.log(P / np.kron(w_s, w_t)) * P).sum()
    return c


def _scaling(
        C: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        epsilon: float,
        tolerance: float = 1e-1,
        max_iteration: int = 10
) -> np.ndarray:
    """
    Algorithm 2 in paper
    :param C: cost matrix c_p or c_q, size (n p)
    :param w_s: source distribution vector, size (n, )
    :param w_t: target distribution vector, size (p, )
    :param rho: float hyperparameter in R+
    :param epsilon: float hyperparameter in R+
    :param tolerance: float parameter to stop process
    :param max_iteration: int max algorithm iteration
    :return:
    """

    n = w_s.shape[0]
    p = w_t.shape[1]

    f = np.zeros(n)
    g = np.zeros(p)
    index = 0
    last_f = None
    last_g = None

    while index < max_iteration:
        tmp_f = 0
        for j in range(p):
            tmp_f += np.exp(
                g[j] + np.log(w_t.reshape(-1, 1)[j]) - (C[:, j] / epsilon)
            )

        f = -(rho/(rho+epsilon)) * np.log(tmp_f)
        tmp_g = 0

        for i in range(n):
            tmp_g += np.exp(
                f[i] + np.log(w_s[i]) - (C[i, :] / epsilon)
            )
        g = -(rho/(rho+epsilon)) * np.log(tmp_g)
        if index != 0:
            if np.linalg.norm(last_f - f) < tolerance and np.linalg.norm(last_g - g) < tolerance:
                P = (np.kron(w_s, w_t.reshape(1, -1)) * np.exp((np.add.outer(f, g) - C / epsilon)))
                return P
        index += 1

        last_f = f
        last_g = g

    P = (np.kron(w_s, w_t.reshape(1, -1)) * np.exp((np.add.outer(f, g) - C / epsilon)))
    return P


def LB_FUGW(
        cost: np.ndarray,
        distance: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        alpha: float,
        epsilon: float,
        max_iteration: int = 50,
        tolerance: float = 1e-1,
        return_i: bool = False
) -> tuple:
    """
    LB Fused-Unbalance-Gromov-Wasserstein algorithm
    :param cost: initial cost matrix, size (n, p)
    :param distance: distance matrix, size (n, p, n, p)
    :param w_s: source distribution vector, size (n, )
    :param w_t: target distribution vector, size (p, )
    :param rho: float hyperparameter in R+
    :param alpha: float hyperparameter in [0, 1]
    :param epsilon: float hyperparameter in R+
    :param max_iteration: int max algorithm iteration
    :param tolerance: float parameter to stop process
    :param return_i: bool to return or not nb iteration
    :rtype: (P, Q) or (P, Q, i)
    :return: Transport and geometry matrix (and nb iteration)
    """
    Q = np.kron(w_s, w_t) / np.sqrt(np.sum(w_s) * np.sum(w_t))
    P = Q
    i = 0
    last_P = None
    last_Q = None
    while i < max_iteration:
        c_p = _cost(
            P=P,
            G=distance,
            C=cost,
            w_s=w_s,
            w_t=w_t,
            rho=rho,
            alpha=alpha,
            epsilon=epsilon
        )

        Q = _scaling(
            C=c_p,
            w_s=w_s,
            w_t=w_t,
            rho=rho * np.sum(P),
            epsilon=epsilon * np.sum(P)
        )

        Q = np.sqrt(np.sum(P) / np.sum(Q)) * Q

        c_q = _cost(
            P=Q,
            G=distance,
            C=cost,
            w_s=w_s,
            w_t=w_t,
            rho=rho,
            alpha=alpha,
            epsilon=epsilon
        )

        P = _scaling(
            C=c_q,
            w_s=w_s,
            w_t=w_t,
            rho=rho * np.sum(Q),
            epsilon=epsilon * np.sum(Q)
        )

        P = np.sqrt(np.sum(Q) / np.sum(P)) * P

        if i != 0:
            if np.linalg.norm(P - last_P) < tolerance and np.linalg.norm(Q - last_Q) < tolerance:
                print(i)
                return (P, Q, i) if return_i else (P, Q)

        last_P = P
        last_Q = Q
        i += 1
    print(i)
    return (P, Q, i) if return_i else (P, Q)
