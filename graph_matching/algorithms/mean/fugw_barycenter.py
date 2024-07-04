"""This module contains barycenter script of FUGW
Implementation of Thual, A., Tran, Q. H., Zemskova, T., Courty, N., Flamary, R. et al.
Aligning individual brains with fused unbalanced Gromov Wasserstein
Advances in neural information processing systems, 35, 21792-21804

.. moduleauthor:: Marius Thorre
"""

import numpy as np
import networkx as nx
import concurrent.futures
from sklearn.cluster import KMeans
import graph_matching.algorithms.pairwise.fugw as fugw


def compute(
        graphs: list,
        max_iteration: int = 10,
        convergence: float = 1e-2
) -> tuple:
    """
    Compute FUGW Barycenter
    :param graphs: nx.Graph's list
    :param max_iteration: maximum loop iteration
    :param convergence: parameter to stop algorithm
    :return: F_b and D_b barycenter
    """
    F_b = _get_init_graph(graphs=graphs)
    D_b = np.random.randint(0, 2, size=(30, 30))

    i = 0
    last_D_b = None
    last_F_b = None

    while i < max_iteration:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fugw_pairwise, g, F_b, D_b) for g in graphs]
            P_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        tmp_f_b = np.zeros(shape=(30, 3))
        tmp_d_b = np.zeros(shape=(30, 30))

        for p in P_list:
            tmp_f_b += np.diag(1 / np.sum(p, axis=0)) @ p.T @ F_b
            tmp_d_b += (p.T @ D_b @ p) / (np.sum(p, axis=0) @ np.sum(p, axis=0).T)
        F_b = (tmp_f_b / len(graphs))
        D_b = (tmp_d_b / len(graphs))

        if i != 0:
            if np.linalg.norm(last_D_b - D_b) < convergence and np.linalg.norm(last_F_b - F_b) < convergence:
                return F_b, D_b

        last_F_b = F_b
        last_D_b = D_b
        i += 1
        print(i)
    return F_b, D_b


def fugw_pairwise(
        g: nx.Graph,
        F_b: np.ndarray,
        D_b: np.ndarray
) -> np.ndarray:
    g_nodes = []
    g_adj = nx.adjacency_matrix(g).todense()
    for index in range(len(g.nodes)):
        if len(g.nodes[index]) > 0:
            g_nodes.append(g.nodes[index]["coord"])
    g_nodes = np.array(g_nodes)
    g_nodes = g_nodes.reshape(g_nodes.shape[1], g_nodes.shape[0]) / 100

    cost = F_b @ g_nodes

    distance = fugw._geometry_cost(g_adj, D_b)
    w_s = np.ones(shape=(30, 1))
    w_t = np.ones(shape=(1, 30))

    P, _ = fugw.LB_FUGW(
        cost=cost,
        distance=distance,
        w_s=w_s,
        w_t=w_t,
        rho=1,
        alpha=0.5,
        epsilon=500,
        tolerance=1e-1
    )

    return P


def _get_init_graph(
        graphs: list,
        nb_cluster: int = 30
):
    coords = [g.nodes[i]["coord"] for g in graphs for i in range(len(g.nodes))]
    kmeans = KMeans(n_clusters=nb_cluster).fit(coords)
    return kmeans.cluster_centers_



