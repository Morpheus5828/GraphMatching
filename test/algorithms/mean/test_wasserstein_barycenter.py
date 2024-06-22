import os, sys

from unittest import TestCase
import math
import networkx as nx
import numpy as np

from graph_matching.algorithms.mean import wasserstein_barycenter
from graph_matching.utils.display_graph_tools import get_graph_from_pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)




def build_noisy_circular_graph(N=20, mu=0, sigma=0.3, with_noise=False, structure_noise=False, p=None):
    """ Create a noisy circular graph
    """
    g = nx.Graph()
    g.add_nodes_from(list(range(N)))
    for i in range(N):
        noise = float(np.random.normal(mu, sigma, 1))
        if with_noise:
            g.add_node(i, attr_name=math.sin((2 * i * math.pi / N)) + noise)
        else:
            g.add_node(i, attr_name=math.sin(2 * i * math.pi / N))
        g.add_edge(i, i + 1)
        if structure_noise:
            randomint = np.random.randint(0, p)
            if randomint == 0:
                if i <= N - 3:
                    g.add_edge(i, i + 2)
                if i == N - 2:
                    g.add_edge(i, 0)
                if i == N - 1:
                    g.add_edge(i, 1)
    g.add_edge(N, 0)
    noise = float(np.random.normal(mu, sigma, 1))
    if with_noise:
        g.add_node(N, attr_name=math.sin((2 * N * math.pi / N)) + noise)
    else:
        g.add_node(N, attr_name=math.sin(2 * N * math.pi / N))
    return g


class TestWassersteinBarycenter(TestCase):
    def testCompute(self):
        graphs = []
        #for k in range(9):
        #    X0.append(build_noisy_circular_graph(np.random.randint(15, 25), with_noise=True, structure_noise=True, p=3))
        # plt.figure(figsize=(8, 10))
        # for i in range(len(X0)):
        #     plt.subplot(3, 3, i + 1)
        #     g = X0[i]
        #     pos = nx.kamada_kawai_layout(g)
        #     nx.draw(g, pos=pos, node_color=graph_colors(g, vmin=-1, vmax=1), with_labels=False, node_size=100)
        # plt.suptitle('Dataset of noisy graphs. Color indicates the label', fontsize=20)
        # plt.show()
        pickle_path = os.path.join(project_path, "graph_matching/demos/graph_generated/pickle")

        for item in os.listdir(pickle_path):
            item_path = os.path.join(pickle_path, item)
            if os.path.isdir(item_path):
                for graph_file in os.listdir(item_path):
                    graph_path = os.path.join(item_path, graph_file)
                    graph = get_graph_from_pickle(graph_path)
                    graphs.append(graph)

        w = wasserstein_barycenter.Barycenter(
            graphs=graphs,
            size_bary=30,
            find_tresh_inf=0.5,
            find_tresh_step=10,
            find_tresh_sup=10,
            graph_vmax=1,
            graph_vmin=-1,
        )
        # print(w.log.keys())
        # print(w.log["Ms"][0].shape)
        w.plot_middle_graph()




