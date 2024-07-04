import numpy as np
import networkx as nx
import graph_matching.algorithms.pairwise.fugw as fugw


def compute(graphs: list, max_iteration: int = 3, convergence: float = 1e-2) -> tuple:
    F_b = np.ones(shape=(30, 3))
    D_b = np.ones(shape=(30, 30))

    i = 0
    last_D_b = None
    last_F_b = None
    while i < max_iteration:
        P_list = []

        for g in graphs:
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
                epsilon=200,
                convergence=1e-1
            )

            P_list.append(P)

        P_list = np.array(P_list).reshape(30, 30)

        F_b *= 1 / len(graphs) * np.sum(np.diag(np.diag(1 / np.sum(P_list, axis=0))) @ P_list.T)

        D_b *= 1 / len(graphs) * np.sum(
                P_list.T @ D_b @ P_list /
                np.sum(P_list, axis=0) @ np.sum(P_list, axis=0).T
            )
        print(D_b)
        # if i != 0:
        #     print(np.linalg.norm(F_b-last_F_b))

        # last_F_b = F_b
        # last_D_b = D_b
        i += 1

    return F_b, D_b
