"""This module contains a matching method between a pair of graph
Implementation of Thual, A., Tran, Q. H., Zemskova, T., Courty, N., Flamary, R. et al.
Aligning individual brains with fused unbalanced Gromov Wasserstein
Advances in neural information processing systems, 35, 21792-21804

.. moduleauthor:: Marius Thorre
"""

import numpy as np


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
    c = alpha * np.kron(G, P)
    print(c.shape)
    c += (1 - alpha) / 2 * C
    print(c.shape)
    #c += rho * (np.log(P.sum(axis=1) / w_s) * P.sum(axis=1)).sum()
    #c += rho * (np.log(P.sum(axis=0) / w_t) * P.sum(axis=0)).sum()
    #c += epsilon * (np.log(P / np.kron(w_s, w_t)) * P).sum()

    return c


def _scaling(
        C: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        epsilon: float,
        tolerance: float = 1e-4,
        max_iteration=1e2
) -> np.ndarray:
    # TODO add convergence parameter
    f = 0
    g = 0
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

    P = np.kron(w_s, w_t) @ np.exp(f + g - C / epsilon)

    return P


def LB_FUGW(
        cost: np.ndarray,
        distance: np.ndarray,
        w_s: np.ndarray,
        w_t: np.ndarray,
        rho: float,
        alpha: float,
        epsilon: float,
        max_iteration: int = 1e4
) -> tuple:
    P, Q = np.kron(w_s, w_t) / np.sqrt(np.sum(w_s) * np.sum(w_t))
    i = 0
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

        Q = np.sqrt(np.sum(P)/np.sum(Q)) * Q

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
        i+=1

    return P, Q
