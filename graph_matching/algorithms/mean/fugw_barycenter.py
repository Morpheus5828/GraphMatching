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
        convergence: float = 1e-1,
        dimension: int = 2
) -> tuple:
    """
    Compute FUGW Barycenter
    :param graphs: nx.Graph's list
    :param max_iteration: maximum loop iteration
    :param convergence: parameter to stop algorithm
    :return: F_b and D_b barycenter
    """
    F_b = _get_init_graph(graphs=graphs)
    D_b = _add_neighbors_edge(F_b)
    i = 0
    last_D_b = None
    last_F_b = None

    while i < max_iteration:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_fugw_pairwise, g, F_b, D_b) for g in graphs]
            P_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        tmp_f_b = np.zeros(shape=(30, 3))
        tmp_d_b = np.zeros(shape=(30, 30))

        for p in P_list:
            tmp_f_b += np.diag(1 / np.sum(p, axis=0)) @ p.T @ F_b
            tmp_d_b += (p.T @ D_b @ p) / ((np.sum(p, axis=0) @ np.sum(p, axis=0).T))

        F_b = (1 / len(graphs)) * tmp_f_b
        D_b = (1 / len(graphs)) * tmp_d_b

        if i != 0:
            if np.linalg.norm(last_D_b - D_b) < convergence and np.linalg.norm(last_F_b - F_b) < convergence:
                return F_b, D_b

        last_F_b = F_b
        last_D_b = D_b

        i += 1

    return F_b, D_b


def _fugw_pairwise(
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

    cost = []
    for i in F_b:
        for j in g_nodes:
            tmp = 0
            tmp += np.abs(i[0] - j[0])
            tmp += np.abs(i[1] - j[1])
            cost.append(tmp)

    cost = np.array(cost)
    cost = cost.reshape(30, 30)

    distance = fugw._geometry_cost(g_adj, D_b)
    w_s = np.ones(shape=(30, 1)) / 30
    w_t = np.ones(shape=(1, 30)) / 30

    P, Q = fugw.LB_FUGW(
        cost=cost,
        distance=distance,
        w_s=w_s,
        w_t=w_t,
        rho=1e-2,
        alpha=0.9,
        epsilon=1e-2,
    )
    return P


def _get_init_graph(
        graphs: list,
        nb_cluster: int = 30
):
    coords = [g.nodes[i]["coord"] for g in graphs for i in range(len(g.nodes))]
    kmeans = KMeans(n_clusters=nb_cluster).fit(coords)
    return kmeans.cluster_centers_


def _add_neighbors_edge(
        coords: np.ndarray,
        nb_neighbors: int = 4
):
    all_nodes = dict(enumerate(coords, 0))
    edges = np.zeros((coords.shape[0], coords.shape[0]))
    for node in all_nodes.keys():
        distance = {}
        for node2 in all_nodes.keys():
            distance[node2] = np.linalg.norm(all_nodes[node] - all_nodes[node2])
        distance = {k: v for k, v in sorted(distance.items(), key=lambda item: item[1])}
        distance = list(distance.keys())
        for d in range(1, nb_neighbors + 1):
            edges[node, distance[d]] = 1
            edges[distance[d], node] = 1

    return edges











