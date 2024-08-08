"""Example of graph pairwcise transport matrix analyse
.. moduleauthor:: Marius Thorre
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import networkx as nx
import time
import graph_matching.algorithms.pairwise.fgw as fgw
import graph_matching.algorithms.pairwise.fugw as fugw
from graph_matching.utils.graph_processing import get_graph_coord, get_graph_from_pickle, _compute_distance
from graph_matching.algorithms.pairwise.pairwise_tools import _get_gradient, _get_constant

if __name__ == '__main__':
    start = time.time()
    G_source = get_graph_from_pickle(
        os.path.join(
            project_root,
            "resources",
            "graph_for_test",
            "generation",
            "without_outliers",
            "noise_60",
            "graph_00000.gpickle"
        )
    )

    G_dest = get_graph_from_pickle(
        os.path.join(
            project_root,
            "resources",
            "graph_for_test",
            "generation",
            "without_outliers",
            "noise_60",
            "graph_00010.gpickle"
        )
    )
    mu_s = np.ones(nx.number_of_nodes(G_source)) / nx.number_of_nodes(G_source)
    mu_s = mu_s.reshape((-1, 1))
    mu_t = np.ones(nx.number_of_nodes(G_dest)) / nx.number_of_nodes(G_dest)
    mu_t = mu_t.reshape((-1, 1))
    adj_matrix_s = nx.adjacency_matrix(G_source).toarray()
    adj_matrix_t = nx.adjacency_matrix(G_dest).toarray()

    graph_coord_s = get_graph_coord(G_source, nb_dimension=3)
    graph_coord_t = get_graph_coord(G_dest, nb_dimension=3)

    distance = _compute_distance(graph_coord_s, graph_coord_t)
    # compute transport pairwise matrix using Fused Gromov Wasserstein algorithm
    transport2 = fgw.conditional_gradient(
        mu_s=mu_s,
        mu_t=mu_t,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        gamma=50,
        ot_method="sinkhorn",
    )

    rho = 50
    alpha = 1
    epsilon = 1

    # use same cost computation than fgw
    c_C1_C2 = _get_constant(
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        transport=mu_s @ mu_t.T
    )
    cost = _get_gradient(
        c_C1_C2=c_C1_C2,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        transport=mu_s @ mu_t.T
    )
    mu_t = mu_t.reshape((1, -1))
    # compute transport pairwise matrix using Fused Unbalanced Gromov Wasserstein algorithm
    P, _ = fugw.LB_FUGW(
        cost=cost,
        distance=fugw._geometry_cost(adj_matrix_s, adj_matrix_t),
        w_s=mu_s,
        w_t=mu_t,
        rho=rho,
        alpha=alpha,
        epsilon=epsilon
    )
    # Compute euclidian distance between both matrices
    end = time.time()
    print(P)
    print(transport2)
    print("Euclidian distance: ", np.linalg.norm(P - transport2))
    print(f"Traning process: {end - start}")
