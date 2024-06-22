"""This module contains tool for display networkx graph
..moduleauthor:: Marius Thorre
"""

import os
import networkx as nx
from graph_matching.utils.graph_processing import save_as_gpickle
import numpy as np
import plotly.graph_objs as go


class Visualisation:
    def __init__(
            self,
            graph: nx.Graph,
            sphere_radius: float = None,
            title: str = "Graph",
            window_width: int = 1000,
            window_height: int = 1000,
    ):
        """
        Object to visualise a graph.
        :param graph: graph paremeter
        :param sphere_radius: sphere radius parameter
        :param title: title of the graph visualisation
        :param window_width: html window width
        :param window_height: html window height
        """
        self.graph = graph
        self.title = title
        self.window_width = window_width,
        self.window_height = window_height,
        self.radius = sphere_radius
        self.fig = None
        self.points = None
        #self.construct_sphere()

    def transform(self) -> None:
        """
        Transform network graph to an other one.
        Some graph has to have the correct name to define structure
        """
        points = []
        for i in range(len(self.graph.nodes)):
            if "coord" in self.graph.nodes[i].keys():
                points.append(self.graph.nodes[i]["coord"])
            elif "sphere_3dcoords" in self.graph.nodes[i].keys():
                points.append(self.graph.nodes[i]["sphere_3dcoords"])
        self.points = np.array(points)

    def check_point_on_sphere(
            self,
            points: np.ndarray,
            radius: float
    ) -> bool:
        """
        Check if a point is on the sphere.
        :param points: array of all sphere coordinates
        :param radius: radius of the sphere
        :return: if all points are on the sphere
        """
        distances = np.linalg.norm(points, axis=1)

        return np.allclose(distances, radius)

    def construct_sphere(self) -> None:
        """
        Construct sphere using all information in inputs
        """
        self.transform()
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

    def save_as_html(self, path_to_save) -> None:
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        self.fig.write_html(os.path.join(path_to_save, self.title + ".html"))

    def save_as_pickle(self, path_to_save: str) -> None:
        save_as_gpickle(os.path.join(path_to_save, self.title), self.graph)

    def compare_and_save(
        self,
        secondGraph: nx.Graph,
        path_to_save: str
    ) -> None:
        """
        Compare two different graph and plot their coordinates on the sphere.
        :param secondGraph: first graph
        :param path_to_save: second graph
        """

        points2 = []
        for i in range(len(secondGraph.nodes)):
            points2.append(secondGraph.nodes[i]["coord"])
        points2 = np.array(points2)

        x1, y1, z1 = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        x2, y2, z2 = points2[:, 0], points2[:, 1], points2[:, 2]

        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1, mode='markers', marker=dict(size=5, color='red', opacity=0.8)
        ))

        self.fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2, mode='markers', marker=dict(size=5, color='green', opacity=0.8)
        ))

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        sphere_x = self.radius * np.cos(u) * np.sin(v)
        sphere_y = self.radius * np.sin(u) * np.sin(v)
        sphere_z = self.radius * np.cos(v)

        self.fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z, opacity=0.3, colorscale='gray'))

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            title=self.title,
            coloraxis_showscale=False
        )
        self.save_as_html(path_to_save)
