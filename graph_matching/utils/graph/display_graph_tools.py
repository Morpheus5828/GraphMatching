"""This module contains tool for display_graph.ipynb
..moduleauthor:: Marius Thorre
"""
import os.path

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
            sphere_radius: float,
            title: str = "",
            window_width: int = 700,
            window_height: int = 700,
    ):
        self.graph = graph
        self.title = title
        self.window_width = window_width,
        self.window_height = window_height,
        self.radius = sphere_radius
        self.fig = None
        points = []
        for i in range(len(self.graph.nodes)):
            points.append(self.graph.nodes[i]["coord"])
        self.points = np.array(points)

        if not self.verify_points_on_sphere(self.points, self.radius):
            print("Some points are not correctly situated on the sphere.")

        self.construct_sphere()

    def verify_points_on_sphere(self, points: np.ndarray, radius: float) -> bool:
        distances = np.linalg.norm(points, axis=1)
        return np.allclose(distances, radius)

    def construct_sphere(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        self.fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red', opacity=0.8)
        )])

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        sphere_x = self.radius * np.cos(u) * np.sin(v)
        sphere_y = self.radius * np.sin(u) * np.sin(v)
        sphere_z = self.radius * np.cos(v)

        self.fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z, opacity=0.3, colorscale='gray'))

        self.fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            title=self.title
        )

    def display(self):
        pass

    def save_as_html(self, path_to_save):
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        self.fig.write_html(os.path.join(path_to_save, self.title + ".html"))

    def save_as_pickle(self, path_to_save):
        save_figure._as_gpickle(os.path.join(path_to_save, self.title), self.graph)
