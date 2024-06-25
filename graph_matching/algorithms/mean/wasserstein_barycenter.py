"""This module contains a computation of the Wasserstein barycenter
https://pythonot.github.io/auto_examples/gromov/plot_barycenter_fgw.html

.. moduleauthor:: Marius THORRE
"""

import numpy as np
import networkx as nx
import ot.gromov
from scipy.sparse.csgraph import shortest_path
from matplotlib import cm
import matplotlib.colors as mcol
from ot.gromov._gw import fused_gromov_wasserstein
from ot.utils import list_to_array, unif, check_random_state, UndefinedParameter, dist
from ot.backend import get_backend
from ot.gromov._utils import update_feature_matrix, update_square_loss, update_kl_loss
import graph_matching.algorithms.pairwise.fgw as fgw


class Barycenter:
    def __init__(
            self,
            graphs: list,
            nb_node: int,
            title: str = "Barycenter"
    ):
        """
        Compute graphs barycenter
        :param graphs: list of graphs
        :param nb_node: barycenter node number that we want
        :param graph_title: graph title
        """
        self.graphs = graphs
        self.size_bary = nb_node
        self.graph_title = title
        self.F, self.A = None, None

    def compute(self) -> None:
        adj_matrices = []
        nodes_positions = []
        for graph in self.graphs:
            adj_matrices.append(nx.adjacency_matrix(graph).todense())
            nodes = []
            for index in range(len(graph.nodes)):
                nodes.append(graph.nodes[index]["coord"])

            nodes_positions.append(np.array(nodes))
        self.F, self.A = ot.gromov.fgw_barycenters(
            N=3,
            Ys=nodes_positions,
            Cs=adj_matrices,
            alpha=0.5,
            fixed_structure=True,
            init_C=adj_matrices[0]
        )

    def get_graph(self) -> nx.Graph:
        G = nx.from_numpy_array(self.A)
        tmp = nx.Graph()
        for node, i in enumerate(G.nodes):
            tmp.add_node(node, coord=self.F[i], label=node)
        return tmp

    def fwg_barycenter(self, N, Ys, Cs, alpha):
        Cs = list_to_array(*Cs)
        Ys = list_to_array(*Ys)
        arr = [*Cs, *Ys]
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
        p = unif(N, type_as=Cs[0])

        n = get_backend(*arr)
        S = len(Cs)
        lambdas = [1. / S] * S
        d = Ys[0].shape[1]
        C = None

