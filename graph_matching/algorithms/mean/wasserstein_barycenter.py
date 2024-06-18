"""This module contains a computation of the Wasserstein barycenter
https://pythonot.github.io/auto_examples/gromov/plot_barycenter_fgw.html#

.. moduleauthor:: Marius THORRE
"""
from typing import List
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from ot.gromov import fgw_barycenters


class Barycenter:
    def __init__(
            self,
            graphs,
            size_bary: int,
            find_tresh_inf: float,
            find_tresh_sup: float,
            find_tresh_step: int,
            graph_vmin: float,
            graph_vmax: float,
            graph_title: str = "Barycenter"
    ):
        self.graphs = graphs
        self.size_bary = size_bary
        self.graph_title = graph_title
        self.find_tresh_inf = find_tresh_inf
        self.find_tresh_sup = find_tresh_sup
        self.find_tresh_step = find_tresh_step
        self.graph_vmin = graph_vmin
        self.graph_vmax = graph_vmax
        self.A, self.C = self.compute()

    def find_thresh(self):
        dist = []
        search = np.linspace(self.find_tresh_inf, self.find_tresh_sup, self.find_tresh_step)
        for thresh in search:
            Cprime = self.sp_to_adjacency(0, thresh)
            SC = shortest_path(Cprime, method='D')
            SC[SC == float('inf')] = 100
            dist.append(np.linalg.norm(SC - self.C))
        return search[np.argmin(dist)], dist

    def sp_to_adjacency(self, threshinf=0.2, threshsup=1.8):
        H = np.zeros_like(self.C)
        np.fill_diagonal(H, np.diagonal(self.C))
        C = self.C - H
        C = np.minimum(np.maximum(C, threshinf), threshsup)
        C[C == threshsup] = 0
        C[C != 0] = 1
        return C

    def graph_colors(self, nx_graph):
        cnorm = mcol.Normalize(vmin=self.graph_vmin, vmax=self.graph_vmax)
        cpick = cm.ScalarMappable(norm=cnorm, cmap='viridis')
        cpick.set_array([])
        val_map = {}
        for k, v in nx.get_node_attributes(nx_graph, 'attr_name').items():
            val_map[k] = cpick.to_rgba(v)
        colors = []
        for node in nx_graph.nodes():
            colors.append(val_map[node])
        return colors

    def get_attributes(self):
        Cs = [shortest_path(nx.adjacency_matrix(x).todense()) for x in self.graphs]
        ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in self.graphs]
        Ys = [
            np.array([v for (k, v) in nx.get_node_attributes(x, 'attr_name').items()]).reshape(-1, 1)
            for x in self.graphs
        ]
        return Cs, ps, Ys

    def compute(self):
        Cs, ps, Ys = self.get_attributes()
        lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
        A, C, _ = fgw_barycenters(self.size_bary, Ys, Cs, ps, lambdas, alpha=0.95, log=True)
        return A, C

    def get_graph(self):
        print(self.sp_to_adjacency(
                threshinf=0,
                threshsup=self.find_thresh()[0]))
        bary = nx.from_numpy_array(
            self.sp_to_adjacency(
                threshinf=0,
                threshsup=self.find_thresh()[0])
        )
        return bary

    def plot_middle_graph(self):
        bary = self.get_graph()
        for i, v in enumerate(self.A.ravel()):
            bary.add_node(i, attr_name=v)
        pos = nx.kamada_kawai_layout(bary)
        nx.draw(
            bary,
            pos=pos,
            with_labels=True,
            node_color=self.graph_colors(bary)
        )
        plt.suptitle(self.graph_title, fontsize=20)
        plt.show()


