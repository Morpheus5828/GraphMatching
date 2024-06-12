"""This module contains tools to generate barycenter graph from graph list
..Moduleauthor:: Marius Thorre
"""
import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
import networkx as nx
import graph_matching.algorithms.mean.wasserstein_barycenter as mean
from graph_matching.utils.graph.display_graph_tools import Visualisation
from graph_matching.utils.graph.graph_processing import get_graph_from_pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
graph_folder = os.path.join(script_dir, "graph_generated", "pickle")
graphs = []

# update attr_name graph
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

barycenter = mean.Barycenter(
    graphs=graphs,
    size_bary=30,
    find_tresh_inf=0.5,
    find_tresh_sup=100,
    find_tresh_step=100,
    graph_vmin=-1,
    graph_vmax=1
    )

bary_graph = barycenter.get_graph()
Visualisation(
    graph=bary_graph,
    title="Barycenter"
).display()
