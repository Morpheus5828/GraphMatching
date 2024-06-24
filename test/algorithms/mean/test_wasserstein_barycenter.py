import os, sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
import graph_matching.algorithms.mean.wasserstein_barycenter as wasserstein_barycenter
from unittest import TestCase
import math
import networkx as nx
from matplotlib import cm
import matplotlib.colors as mcol
import numpy as np

# def graph_colors(nx_graph, vmin=0, vmax=7):
#     cnorm = mcol.Normalize(vmin=vmin, vmax=vmax)
#     cpick = cm.ScalarMappable(norm=cnorm, cmap='viridis')
#     cpick.set_array([])
#     val_map = {}
#     for k, v in nx.get_node_attributes(nx_graph, 'attr_name').items():
#         val_map[k] = cpick.to_rgba(v)
#     colors = []
#     for node in nx_graph.nodes():
#         colors.append(val_map[node])
#     return colors
#
#
# def build_noisy_circular_graph(N=20, mu=0, sigma=0.3, with_noise=False, structure_noise=False, p=None):
#     """ Create a noisy circular graph
#     """
#     g = nx.Graph()
#     g.add_nodes_from(list(range(N)))
#     for i in range(N):
#         noise = float(np.random.normal(mu, sigma, 1))
#         if with_noise:
#             g.add_node(i, attr_name=math.sin((2 * i * math.pi / N)) + noise)
#         else:
#             g.add_node(i, attr_name=math.sin(2 * i * math.pi / N))
#         g.add_edge(i, i + 1)
#         if structure_noise:
#             randomint = np.random.randint(0, p)
#             if randomint == 0:
#                 if i <= N - 3:
#                     g.add_edge(i, i + 2)
#                 if i == N - 2:
#                     g.add_edge(i, 0)
#                 if i == N - 1:
#                     g.add_edge(i, 1)
#     g.add_edge(N, 0)
#     noise = float(np.random.normal(mu, sigma, 1))
#     if with_noise:
#         g.add_node(N, attr_name=math.sin((2 * N * math.pi / N)) + noise)
#     else:
#         g.add_node(N, attr_name=math.sin(2 * N * math.pi / N))
#     return g
#
#
# class TestWassersteinBarycenter(TestCase):
#     def testCompute(self):
#         X0 = []
#         for k in range(9):
#             X0.append(build_noisy_circular_graph(np.random.randint(15, 25), with_noise=True, structure_noise=True, p=3))
#         # plt.figure(figsize=(8, 10))
#         # for i in range(len(X0)):
#         #     plt.subplot(3, 3, i + 1)
#         #     g = X0[i]
#         #     pos = nx.kamada_kawai_layout(g)
#         #     nx.draw(g, pos=pos, node_color=graph_colors(g, vmin=-1, vmax=1), with_labels=False, node_size=100)
#         # plt.suptitle('Dataset of noisy graphs. Color indicates the label', fontsize=20)
#         # plt.show()
#
#         w = wasserstein_barycenter.Barycenter(
#             graphs=X0,
#             size_bary=15,
#             find_tresh_inf=0.5,
#             find_tresh_step=10,
#             find_tresh_sup=10,
#             graph_vmax=1,
#             graph_vmin=-1
#         )
#
#         w.plot_middle_graph()




