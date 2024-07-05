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
from graph_matching.utils.graph_processing import get_graph_coord, get_graph_from_pickle

if __name__ == '__main__':
    start = time.time()
    G_source = get_graph_from_pickle(
        os.path.join(
            project_root,
            "resources",
            "graph_for_test",
            "generation",
            "without_outliers",
            "noise_1_outliers_varied",
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
            "noise_1_outliers_varied",
            "graph_00001.gpickle"
        )
    )
    mu_s = np.ones(nx.number_of_nodes(G_source)) / nx.number_of_nodes(G_source)
    mu_s = mu_s.reshape((-1, 1))
    mu_t = np.ones(nx.number_of_nodes(G_dest)) / nx.number_of_nodes(G_dest)
    #mu_t = mu_t.reshape((-1, 1))

    adj_matrix_s = nx.adjacency_matrix(G_source).toarray()
    adj_matrix_t = nx.adjacency_matrix(G_dest).toarray()

    graph_coord_s = get_graph_coord(G_source, nb_dimension=3)
    graph_coord_t = get_graph_coord(G_dest, nb_dimension=3)

    distance = fgw._M(graph_coord_s, graph_coord_t)

    # transport1 = fgw.conditional_gradient(
    #     mu_s=mu_s,
    #     mu_t=mu_t,
    #     C1=adj_matrix_s,
    #     C2=adj_matrix_t,
    #     distance=distance,
    #     gamma=0.15,
    #     ot_method="sinkhorn"
    # )
    #
    transport2 = fgw.conditional_gradient(
        mu_s=mu_s,
        mu_t=mu_t,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        rho=80,
        eta=50,
        N1=50,
        N2=50,
        ot_method="sns",
    )
    cost = graph_coord_s @ graph_coord_t.reshape(graph_coord_t.shape[1], graph_coord_t.shape[0])/100
    P, _ = fugw.LB_FUGW(
        cost=cost,
        distance=fugw._geometry_cost(adj_matrix_s, adj_matrix_t),
        w_s=mu_s,
        w_t=mu_t,
        rho=1,
        epsilon=1e5,
        alpha=0.5

    )
    end = time.time()
    print("Distance euclidienne ", np.linalg.norm(P - transport2))
    print(f"Traning process: {end - start}")
