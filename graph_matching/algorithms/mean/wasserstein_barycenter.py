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
from sklearn.neighbors import KNeighborsClassifier


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
        self.nb_node = nb_node
        self.graph_title = title
        self.F, self.A = None, None

    def compute(self, fixed_structure=False) -> None:
        adj_matrices = []
        nodes_positions = []
        nodes_label = []
        for graph in self.graphs:
            adj_matrices.append(nx.adjacency_matrix(graph).todense())
            nodes = []
            label = []
            for index in range(len(graph.nodes)):
                nodes.append(graph.nodes[index]["coord"])
                label.append(graph.nodes[index]["label"])
            nodes_positions.append(np.array(nodes))
            nodes_label.append(np.array(label))
        if fixed_structure:
            self.F, self.A = ot.gromov.fgw_barycenters(
                N=self.nb_node,
                Ys=nodes_positions,
                Cs=adj_matrices,
                alpha=0.5,
                fixed_structure=True,
                init_C=adj_matrices[0]
            )
        else:
            self.F, self.A = ot.gromov.fgw_barycenters(
                N=self.nb_node,
                Ys=nodes_positions,
                Cs=adj_matrices,
                alpha=0.5,
                fixed_structure=True,
                init_C=adj_matrices[0]
            )

    def get_graph(self) -> nx.Graph:
        all_node_coord = []
        all_node_label = []

        for graph in self.graphs:
            for index in range(len(graph.nodes)):
                all_node_coord.append(graph.nodes[index]["coord"])
                all_node_label.append(graph.nodes[index]["label"])

        all_node_coord = np.array(all_node_coord)
        all_node_label = np.array(all_node_label).reshape(-1, 1)

        clf = KNeighborsClassifier(n_neighbors=20)
        clf.fit(all_node_coord, all_node_label)

        G = nx.from_numpy_array(self.A)
        tmp = nx.Graph()

        for node, i in enumerate(G.nodes):
            tmp.add_node(node, coord=self.F[i], label=int(clf.predict(self.F[i].reshape(1, -1))))
        return tmp

    # def fwg_barycenter(self, N, Ys, Cs, alpha):
    #     Cs = list_to_array(*Cs)
    #     Ys = list_to_array(*Ys)
    #     arr = [*Cs, *Ys]
    #     ps = [unif(C.shape[0], type_as=C) for C in Cs]
    #     p = unif(N, type_as=Cs[0])
    #
    #     n = get_backend(*arr)
    #     S = len(Cs)
    #     lambdas = [1. / S] * S
    #     d = Ys[0].shape[1]
    #     C = None

