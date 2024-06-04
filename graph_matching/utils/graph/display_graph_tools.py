"""This module contains tool for display_graph.ipynb
..moduleauthor:: Marius Thorre
"""
import networkx as nx
import pickle
import igraph as ig
import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go


def get_graph_from_pickle(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph


class Visualisation:
    def __init__(
            self,
            graph: nx.Graph,
            title: str = "",
            path_to_save: str = "graph_matching/demos/graph/",
            window_width: int = 1000,
            window_height: int = 1000
    ):
        self.graph = graph
        self.Xn = 0
        self.Yn = 0
        self.Zn = 0
        self.Xe = 0
        self.Ye = 0
        self.Ze = 0
        self.title = title
        self.window_width = window_width,
        self.window_height = window_height,
        self.fig = None
        self.path_to_save = path_to_save

        self.configure_layout_param()
        data = self.configure_trace()
        layout = self.configure_layout()
        self.fig = go.Figure(data=data, layout=layout)

    def get_graph_coord(
            self,
            graph: nx.Graph,
            nb_dimension: int
    ) -> np.ndarray:
        graph_coord = np.zeros(shape=(nx.number_of_nodes(graph), nb_dimension))
        for node in graph.nodes(data=True):
            graph_coord[node[0]] = node[1]["coord"]

        return graph_coord

    def configure_layout_param(self):
        nb_node = len(self.graph.nodes)
        G = ig.Graph(self.graph.edges, directed=False)
        graph_layout = G.layout('kk', dim=3)

        self.Xn = [graph_layout[k][0] for k in range(nb_node)]  # x-coordinates of nodes
        self.Yn = [graph_layout[k][1] for k in range(nb_node)]  # y-coordinates
        self.Zn = [graph_layout[k][2] for k in range(nb_node)]  # z-coordinates
        self.Xe = []
        self.Ye = []
        self.Ze = []
        for e in self.graph.edges:
            self.Xe += [graph_layout[e[0]][0], graph_layout[e[1]][0], None]  # x-coordinates of edge ends
            self.Ye += [graph_layout[e[0]][1], graph_layout[e[1]][1], None]
            self.Ze += [graph_layout[e[0]][2], graph_layout[e[1]][2], None]

    def configure_trace(self):
        trace1 = go.Scatter3d(
            x=self.Xe,
            y=self.Ye,
            z=self.Ze,
            mode='lines',
            line=dict(color='rgb(125,125,125)', width=1),
            hoverinfo='none'
        )
        trace2 = go.Scatter3d(
            x=self.Xn,
            y=self.Yn,
            z=self.Zn
        )
        return trace1, trace2

    def configure_layout(self):
        axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=""
        )

        layout = go.Layout(
            title=self.title,
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest'
        )

        return layout

    def display(self):
        iplot(self.fig)

    def save_as_html(self):
        self.fig.write_html("C:/Users/thorr/PycharmProjects/GraphMatching/graph_matching/demos/" + self.title + ".html")

