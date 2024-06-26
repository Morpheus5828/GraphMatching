"""This module contains tool for display networkx graph
..moduleauthor:: Marius Thorre
"""

import os
import networkx as nx
from graph_matching.utils.graph_processing import save_as_gpickle, get_graph_from_pickle
import numpy as np
import plotly.graph_objs as go


class Visualisation:
    def __init__(
            self,
            graph: nx.Graph = None,
            sphere_radius: float = 90,
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
        self.fig = go.Figure()
        self.points = None
        self.labels = None
        self.all_color = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Brown', 'Black', 'White',
                          'Gray', 'Violet', 'Cyan', 'Magenta', 'Lime', 'Maroon', 'Olive', 'Navy', 'Teal', 'Aqua',
                          'Coral', 'Turquoise', 'Beige', 'Lavender', 'Salmon', 'Gold', 'Silver', 'aliceblue', 'Khaki',
                          'Indigo']

    def transform(self) -> None:
        """
        Transform network graph to another one.
        Some graph has to have the correct name to define structure
        """
        points = []
        labels = []
        for i in range(len(self.graph.nodes)):
            if "coord" in self.graph.nodes[i].keys():
                points.append(self.graph.nodes[i]["coord"])
            elif "sphere_3dcoords" in self.graph.nodes[i].keys():
                points.append(self.graph.nodes[i]["sphere_3dcoords"])
            labels.append(self.graph.nodes[i]["label"])
        self.points = np.array(points)
        self.labels = np.array(labels)

    def check_point_on_sphere(self, points: np.ndarray, radius: float) -> bool:
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
        current_color = [self.all_color[i] if i != -1 else "Crimson" for i in self.labels]

        self.fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=current_color, opacity=0.8)
        )])

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        sphere_x = self.radius * np.cos(u) * np.sin(v)
        sphere_y = self.radius * np.sin(u) * np.sin(v)
        sphere_z = self.radius * np.cos(v)


        self.fig.add_trace(
            go.Surface(
                x=sphere_x,
                y=sphere_y,
                z=sphere_z,
                opacity=0.3,
                colorscale='gray',
                showscale=False
            )
        )

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            title=f"Graph name: {self.title}",
            showlegend=True,
            annotations=[
                dict(
                    showarrow=False,
                    text=f"Nb nodes: {len(self.graph.nodes)} <br>Nb edge: {len(self.graph.edges)}",
                    xref="paper",
                    yref="paper",
                    x=0.95,
                    y=0.5,
                    xanchor='right',
                    yanchor='middle',
                    font=dict(size=16, color="black")
                )
            ]
        )

    def save_as_html(self, path_to_save: str) -> None:
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        self.fig.write_html(os.path.join(path_to_save, self.title + ".html"))

    def save_as_pickle(self, path_to_save: str) -> None:
        save_as_gpickle(os.path.join(path_to_save, self.title), self.graph)

    def add_graph_to_plot(self, second_graph: nx.Graph, radius=90):
        coord = []
        label = []
        for i in range(len(second_graph.nodes)):
            if len(second_graph.nodes[i]) != 0:
                coord.append(second_graph.nodes[i]["coord"])
                label.append(second_graph.nodes[i]["label"])

        current_color = [self.all_color[i] if i != -1 else "Crimson" for i in label]
        coord = np.array(coord)
        x1, y1, z1 = coord[:, 0], coord[:, 1], coord[:, 2]

        self.fig.add_trace(go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            mode='markers',
            marker=dict(size=5, color=current_color
                        , opacity=0.8),
            showlegend=True
        ))

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        sphere_x = radius * np.cos(u) * np.sin(v)
        sphere_y = radius * np.sin(u) * np.sin(v)
        sphere_z = radius * np.cos(v)

        self.fig.add_trace(go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.3,
            colorscale='gray',
            showscale=False,
            showlegend=False
        ))

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=True
        )

    def plot_graphs(
            self,
            folder_path: str,
            radius=90
    ) -> None:

        for graph in os.listdir(folder_path):
            graph = get_graph_from_pickle(os.path.join(folder_path, graph))
            coord = []
            label = []
            for i in range(len(graph.nodes)):
                if len(graph.nodes[i]) != 0:
                    coord.append(graph.nodes[i]["coord"])
                    label.append(graph.nodes[i]["label"])

            current_color = [self.all_color[i] if i != -1 else "Crimson" for i in label]
            coord = np.array(coord)
            x1, y1, z1 = coord[:, 0], coord[:, 1], coord[:, 2]

            self.fig.add_trace(go.Scatter3d(
                x=x1,
                y=y1,
                z=z1,
                mode='markers',
                marker=dict(size=5, color=current_color
                            , opacity=0.8),
                showlegend=True
            ))

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        sphere_x = radius * np.cos(u) * np.sin(v)
        sphere_y = radius * np.sin(u) * np.sin(v)
        sphere_z = radius * np.cos(v)

        self.fig.add_trace(go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.3,
            colorscale='gray',
            showscale=False,
            showlegend=False
        ))

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=True
        )
