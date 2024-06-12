"""This module contains tool for display_graph.ipynb
..moduleauthor:: Marius Thorre
"""
import os.path

import matplotlib.pyplot as plt
import networkx as nx
import pickle
import graph_matching.utils.pickle.save_figure as save_figure
import igraph as ig
import numpy as np
from plotly.offline import iplot
import graph_matching.utils.color_generation as color_generation
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
            window_width: int = 700,
            window_height: int = 700
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

        self.Xn = [graph_layout[k][0] for k in range(nb_node)]
        self.Yn = [graph_layout[k][1] for k in range(nb_node)]
        self.Zn = [graph_layout[k][2] for k in range(nb_node)]
        self.Xe = []
        self.Ye = []
        self.Ze = []
        for e in self.graph.edges:
            self.Xe += [graph_layout[e[0]][0], graph_layout[e[1]][0], None]
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
            z=self.Zn,
            text=np.arange(len(self.graph.nodes))
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
            width=800,
            height=800,
            showlegend=True,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            hovermode='closest'
        )

        return layout

    def display(self):
        colors = []
        for _ in range(len(self.graph.nodes)):
            colors.append(color_generation.generate_new_color(colors))
        nx.draw(self.graph, with_labels=True, node_color=colors)
        plt.show()
        self.fig.update_layout(title_text=f"Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}")
        iplot(self.fig)

    def save_as_html(self, path_to_save):
        self.fig.update_layout(title_text=f"Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}")
        self.fig.write_html(os.path.join(path_to_save, self.title + ".html"))

    def save_as_pickle(self, path_to_save):
        save_figure._as_gpickle(os.path.join(path_to_save, self.title), self.graph)
