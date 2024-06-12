import os.path
from unittest import TestCase

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import graph_matching.algorithms.mean.wasserstein_barycenter as mean
from graph_matching.utils.graph.display_graph_tools import Visualisation
from graph_matching.utils.graph.graph_processing import get_graph_coord, get_graph_from_pickle


class TestWassersteinBarycenter(TestCase):
    def test_fgw_wasserstein_barycenter(self):
        # g2 = nx.Graph()
        # g2.add_node(0, weight=np.array((5.0,)))
        # g2.add_node(1, weight=np.array((1.0,)))
        # g2.add_node(2, weight=np.array((2.0,)))
        # g2.add_node(3, weight=np.array((0.0,)))
        # g2.add_edge(0, 1, weight=1.0)
        # g2.add_edge(0, 2, weight=4.0)

        graph_folder = "C:/Users/thorr/PycharmProjects/GraphMatching/graph_matching/demos/graph_generated/pickle/"
        graphs = []


        for file in os.listdir(graph_folder):
            if os.path.isdir(os.path.join(graph_folder, file)):
                for graph in os.listdir(os.path.join(graph_folder, file)):
                    g = get_graph_from_pickle(os.path.join(graph_folder, file, graph))
                    tmp = nx.Graph()
                    tmp.add_nodes_from(list(range(len(g.nodes))))
                    for node in g.nodes:
                        tmp.add_node(node, attr_name=node)

                    for edge in g.edges:
                        tmp.add_edge(edge[0], edge[1])
                    graphs.append(tmp)


        G1 = get_graph_from_pickle("C:/Users/thorr/PycharmProjects/GraphMatching/graph_matching/demos/graph_generated/pickle/reference.gpickle")
        print(len(G1.nodes))
        tmp = nx.Graph()
        tmp.add_nodes_from(list(range(len(G1.nodes))))
        for node in G1.nodes:
            tmp.add_node(node, attr_name=node)
        for edge in G1.edges:
            tmp.add_edge(edge[0], edge[1])


        b = mean.Barycenter(
            graphs=[tmp, tmp],
            size_bary=30,
            find_tresh_inf=0.5,
            find_tresh_sup=100,
            find_tresh_step=100,
            graph_vmin=-1,
            graph_vmax=1
        )

        a = b.get_graph()
        # nx.draw(a)
        # plt.show()

        v = Visualisation(
            graph=a,
            title="barycenter",
        )

        v.save_as_html(path_to_save="C:/Users/thorr/PycharmProjects/GraphMatching/graph_matching/demos")


