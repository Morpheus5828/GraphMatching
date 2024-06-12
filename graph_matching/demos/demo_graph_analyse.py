"""This module contains tool to compare graph and compute transport matrix
..Moduleauthor:: Marius Thorre
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import networkx as nx
import time
import graph_matching.algorithms.pairwise.fgw as fgw
from graph_matching.utils.graph.graph_processing import get_graph_coord, get_graph_from_pickle


if __name__ == '__main__':
    start = time.time()

    G_source = get_graph_from_pickle("graph_matching/demos/graph_generated/0/reference_0.gpickle", )
    G_dest = get_graph_from_pickle("graph_matching/demos/graph_generated/1/reference_1.gpickle")

    mu_s = np.ones(nx.number_of_nodes(G_source)) / nx.number_of_nodes(G_source)
    mu_s = mu_s.reshape((-1, 1))
    mu_t = np.ones(nx.number_of_nodes(G_dest)) / nx.number_of_nodes(G_dest)
    mu_t = mu_t.reshape((-1, 1))

    adj_matrix_s = nx.adjacency_matrix(G_source).toarray()
    adj_matrix_t = nx.adjacency_matrix(G_dest).toarray()

    graph_coord_s = get_graph_coord(G_source, nb_dimension=3)
    graph_coord_t = get_graph_coord(G_dest, nb_dimension=3)

    distance = fgw._M(graph_coord_s, graph_coord_t)

    transport1 = fgw.conditional_gradient(
        mu_s=mu_s,
        mu_t=mu_t,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        gamma=1/2,
        tolerance=0.1,
        ot_method="sinkhorn"
    )

    transport2 = fgw.conditional_gradient(
        mu_s=mu_s,
        mu_t=mu_t,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        rho=2,
        eta=15,
        N1=50,
        N2=50,
        tolerance=0.1,
        ot_method="sns",
    )

    transport3 = fgw.conditional_gradient(
        mu_s=mu_s,
        mu_t=mu_t,
        C1=adj_matrix_s,
        C2=adj_matrix_t,
        distance=distance,
        rho=2,
        eta=15,
        N1=50,
        N2=50,
        tolerance=0.1,
        ot_method="fx_sns",
    )
    end = time.time()

    print("Distance euclidienne ", np.linalg.norm(transport1 - transport2))
    print(f"Traning process: {end - start}")


