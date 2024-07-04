"""This module contains a matching method between a pair of graph
Implementation of Thual, A., Tran, Q. H., Zemskova, T., Courty, N., Flamary, R. et al.
Aligning individual brains with fused unbalanced Gromov Wasserstein
Advances in neural information processing systems, 35, 21792-21804

.. moduleauthor:: Marius Thorre
"""

import numpy as np


def _geometry_cost(D_s, D_t):
    n = D_s.shape[0]
    p = D_t.shape[0]

    result = []
    for i in range(n):
        for j in range(p):
            for k in range(n):
                for l in range(p):
                    result.append(np.abs(D_s[i, k] - D_t[j, l]))

    return np.array(result).reshape(n, p, n, p)


def _kron(G, P):
    K = G.shape[2]
    L = G.shape[3]

    result = []
    for k in range(K):
        for l in range(L):
            result.append(
                np.sum(G, axis=(2, 3))[k, l] * np.sum(P)
            )
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
    c = alpha * _kron(G, P)
    c += (1 - alpha) / 2 * C
    c += rho * (np.log(P.sum(axis=0) / w_s) * P.sum(axis=0)).sum()
    c += rho * (np.log(P.sum(axis=1) / w_s) * P.sum(axis=1)).sum()
    c += epsilon * (np.log(P / np.kron(w_s, w_t)) * P).sum()
    return c

# TODO add convergence parameter
def _scaling(
        C: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        epsilon: float,
        tolerance: float = 1e-4,
        max_iteration=10
) -> np.ndarray:

    f = np.zeros(shape=w_s.shape)
    g = np.zeros(shape=w_t.shape)
    i = 0
    while i < max_iteration:
        f_next = -(rho / rho + epsilon)
        f_next *= np.log(
            np.sum(
                np.exp(
                    g + np.log(w_t) - C / epsilon
                )
            )
        )

        g_next = -(rho / rho + epsilon)
        g_next *= np.log(
            np.sum(
                np.exp(
                    f + np.log(w_s) - C / epsilon
                )
            )
        )

        i += 1

    g = g.reshape(1, -1)
    P = np.kron(w_s, w_t) * np.exp(f + g - C / epsilon)
    return P


def LB_FUGW(
        cost: np.ndarray,
        distance: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        alpha: float,
        epsilon: float,
        max_iteration: int = 10,
        convergence: float = 1e-2,
        return_i: bool = False
) -> tuple:
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

        Q = _scaling(
            C=c_q,
            w_s=w_s,
            w_t=w_t,
            rho=rho * np.sum(Q),
            epsilon=epsilon * np.sum(Q)
        )

        P = np.sqrt(np.sum(Q) / np.sum(P)) * P
        if i != 0:
            if np.linalg.norm(P-last_P) < convergence and np.linalg.norm(Q-last_Q) < convergence:
                return (P, Q, i) if return_i else (P, Q)

        last_P = P
        last_Q = Q
        i += 1

    return (P, Q, i) if return_i else (P, Q)
