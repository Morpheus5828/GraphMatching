"""This module contains barycenter script of FUGW
Implementation of Thual, A., Tran, Q. H., Zemskova, T., Courty, N., Flamary, R. et al.
Aligning individual brains with fused unbalanced Gromov Wasserstein
Advances in neural information processing systems, 35, 21792-21804

.. moduleauthor:: Marius Thorre
"""

import random
import numpy as np
import networkx as nx
import concurrent.futures
from sklearn.cluster import KMeans
import graph_matching.algorithms.pairwise.fugw as fugw
import graph_matching.algorithms.pairwise.fgw as fgw
from graph_matching.utils.graph_processing import _compute_distance

def compute(
        graphs: list,
        alpha: float, epsilon: float, rho: float,
        max_iteration: int = 50,
        convergence: float = 1e-1,
) -> tuple:
    """
    Compute FUGW Barycenter
    :param alpha:
    :param epsilon:
    :param rho:
    :param graphs: nx.Graph's list
    :param max_iteration: maximum loop iteration
    :param convergence: parameter to stop algorithm
    :return: F_b and D_b barycenter
    """

    F_b, sample_graphe = _get_init_graph(graphs=graphs)
    F_b /= 100
    D_b = np.zeros((30, 30))

    i = 0
    last_F_b = None
    last_D_b = None

    while i < max_iteration:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_fugw_pairwise, g, F_b, D_b, alpha, epsilon, rho) for g in sample_graphe]
            p_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        tmp_F_b = np.zeros((30, 3))
        tmp_d_b = np.zeros((30, 30))
        for p in p_list:
            tmp_F_b += np.diag(1 / (np.sum(p, axis=0))) @ p.T @ F_b
            tmp_d_b += (p.T @ D_b @ p) / (np.sum(p, axis=0) @ np.sum(p, axis=0).T)

        F_b = (1 / len(sample_graphe)) * tmp_F_b
        D_b = (tmp_d_b / len(graphs))

        if i != 0:
            if np.linalg.norm(last_F_b - F_b) < convergence and np.linalg.norm(last_D_b - D_b) < convergence:
                return F_b, D_b

        last_F_b = F_b
        last_D_b = D_b
        i += 1
    return F_b, D_b


def _fugw_pairwise(
        g: nx.Graph,
        F_b: np.ndarray,
        D_b: np.ndarray,
        alpha: float, epsilon: float, rho: float,
) -> np.ndarray:
    g_nodes = []
    g_adj = nx.adjacency_matrix(g).todense()
    for index in range(len(g.nodes)):
        if len(g.nodes[index]) > 0:
            g_nodes.append(g.nodes[index]["coord"])
    g_nodes = np.array(g_nodes) / 100

    distance = []
    for i in g_nodes:
        for j in F_b:
            tmp = 0
            tmp += np.abs(i[0] - j[0])
            tmp += np.abs(i[1] - j[1])
            tmp += np.abs(i[2] - j[2])
            distance.append(tmp)

    distance = np.array(distance).reshape(30, 30)

    w_s = np.ones(shape=(30, 1)) / 30
    w_t = np.ones(shape=(1, 30)) / 30

    c_src_dest = fgw._get_constant(
        C1=g_adj,
        C2=D_b,
        distance=distance,
        transport=w_s @ w_t
    )
    cost = fgw._get_gradient(
        c_C1_C2=c_src_dest,
        C1=g_adj,
        C2=D_b,
        distance=distance,
        transport=w_s @ w_t
    )

    P, _ = fugw.LB_FUGW(
        cost=cost,
        distance=fugw._geometry_cost(g_adj, D_b),
        w_s=w_s,
        w_t=w_t,
        rho=rho,
        alpha=alpha,
        epsilon=epsilon
    )

    return P


def _get_init_graph(
        graphs: list,
):
    index = random.randint(0, len(graphs) - 1)
    coord = [graphs[index].nodes[i]["coord"] for i in range(len(graphs[index].nodes))]
    sample_graph = [g for i, g in enumerate(graphs) if i != index]
    return np.array(coord), sample_graph


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


def get_graph(graphs: list, rho: float, epsilon: float, alpha: float):
    F_b, _ = compute(
        graphs=graphs,
        rho=rho,
        epsilon=epsilon,
        alpha=alpha
    )
    F_b *= 100
    return F_b[:4]
