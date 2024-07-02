"""This module contains tool for display networkx graph
..moduleauthor:: Marius Thorre
"""

import os
import sys
from tqdm import tqdm
import networkx as nx
from graph_matching.utils.graph_processing import save_as_gpickle, get_graph_from_pickle, check_point_on_sphere
import numpy as np
import plotly.graph_objs as go
import nibabel as nib


class Visualisation:
    def __init__(
            self,
            graph: nx.Graph = None,
            sphere_radius: float = 100,
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
        self.all_color = [
            'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Brown', 'Black', 'White',
            'Gray', 'Violet', 'Cyan', 'Magenta', 'Lime', 'Maroon', 'Olive', 'Navy', 'Teal', 'Aqua',
            'Coral', 'Turquoise', 'Beige', 'Lavender', 'Salmon', 'Gold', 'Silver', 'aliceblue', 'Khaki',
            'Indigo', 'Plum'
        ]
        if self.graph is not None:
            self.extract_coord_label()
            self.construct_sphere()

    def extract_coord_label(self) -> None:
        """
        Transform network graph to another one.
        Some graph has to have the correct name to define structure
        """
        points = []
        labels = []
        for i in range(len(self.graph.nodes)):
            if len(self.graph.nodes[i]) != 0:
                if "coord" in self.graph.nodes[i].keys():
                    points.append(self.graph.nodes[i]["coord"])
                elif "sphere_3dcoords" in self.graph.nodes[i].keys():
                    points.append(self.graph.nodes[i]["sphere_3dcoords"])
                labels.append(self.graph.nodes[i]["label"])
        self.points = np.array(points)
        self.labels = np.array(labels)

    def construct_sphere(self) -> None:
        """
        Construct sphere using all information in inputs
        """
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        current_color = [self.all_color[i] if i != 0 and i != -1 else ("Crimson" if i == 0 else "amber") for i in self.labels]

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

    def add_graph_to_plot(self, second_graph: nx.Graph, radius=100):
        coord = []
        label = []
        for i in range(len(second_graph.nodes)):
            if len(second_graph.nodes[i]) != 0:
                coord.append(second_graph.nodes[i]["coord"])
                label.append(int(second_graph.nodes[i]["label"]))

        current_color = [self.all_color[i] if i != 0 and i != -1 else ("Crimson" if i == 0 else "amber") for i in label]
        coord = np.array(coord)
        x1, y1, z1 = coord[:, 0], coord[:, 1], coord[:, 2]

        self.fig.add_trace(go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            mode='markers',
            marker=dict(
                size=5,
                color=current_color,
                opacity=0.8
            ),
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

    def plot_graph_on_mesh(
            self,
            cortext_mesh_path: str,
            sphere_mesh_path: str,
    ) -> None:
        self.extract_coord_label()
        sphere_mesh = nib.load(sphere_mesh_path)
        sphere_vertices = sphere_mesh.darrays[0].data.astype(float)

        cortex_mesh = nib.load(cortext_mesh_path)
        cortex_vertices = cortex_mesh.darrays[0].data.astype(float)
        coord_to_plot = []
        for i in range(len(self.points)):
            x, y, z = self.points[i, 0], self.points[i, 1], self.points[i, 2]
            tmp_dist = sys.maxsize
            tmp_index = 0

            sphere_vertices = np.array(sphere_vertices)

            for j, coord in enumerate(sphere_vertices):
                if np.linalg.norm(coord - np.array([x, y, z])) < tmp_dist:
                    tmp_index = j
                    tmp_dist = np.linalg.norm(coord - np.array([x, y, z]))

            x, y, z = sphere_vertices[tmp_index]

            tmp_dist = sys.maxsize
            tmp_index = 0
            for j, coord in enumerate(cortex_vertices):
                if np.linalg.norm(coord - np.array([x, y, z])) < tmp_dist:
                    tmp_index = j
                    tmp_dist = np.linalg.norm(coord - np.array([x, y, z]))

            coord_to_plot.append(cortex_vertices[tmp_index])

        coord_to_plot = np.array(coord_to_plot)

        current_color = [self.all_color[i] if i != 0 and i != -1 else ("Crimson" if i == 0 else "amber") for i in self.labels]
        scatter = go.Scatter3d(
            x=coord_to_plot[:, 0],
            y=coord_to_plot[:, 1],
            z=coord_to_plot[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=current_color,
                opacity=0.8)
        )

        vertices = cortex_mesh.darrays[0].data.astype(float)
        faces = cortex_mesh.darrays[1].data.astype(int)
        vertices[:, 1] *= -1
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='gray',
            opacity=0.50
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=True,
            title="Cortex Mesh"
        )

        fig = go.Figure(data=[scatter, mesh], layout=layout)

        fig.update_layout(scene=dict(aspectmode='data'))

        fig.show()

    def plot_all_graph_on_mesh(
            self,
            folder_path: str,
            cortext_mesh_path: str,
            sphere_mesh_path: str,
    ) -> None:
        graphs = []
        for graph in os.listdir(folder_path):
            graphs.append(get_graph_from_pickle(os.path.join(folder_path, graph)))

        sphere_mesh = nib.load(sphere_mesh_path)
        sphere_vertices = sphere_mesh.darrays[0].data.astype(float)

        cortex_mesh = nib.load(cortext_mesh_path)
        cortex_vertices = cortex_mesh.darrays[0].data.astype(float)

        scatters = []

        for i in tqdm(range(len(graphs))):
            g = graphs[i]
            points = []
            label = []
            for i in range(len(g.nodes)):
                if len(g.nodes[i]) != 0:
                    if "coord" in g.nodes[i].keys():
                        points.append(g.nodes[i]["coord"])
                    elif "sphere_3dcoords" in g.nodes[i].keys():
                        points.append(g.nodes[i]["sphere_3dcoords"])
                    label.append(g.nodes[i]["label"])
            points = np.array(points)
            coord_to_plot = []
            for i in range(len(points)):
                x, y, z = points[i, 0], points[i, 1], points[i, 2]
                tmp_dist = sys.maxsize
                tmp_index = 0

                sphere_vertices = np.array(sphere_vertices)

                for j, coord in enumerate(sphere_vertices):
                    if np.linalg.norm(coord - np.array([x, y, z])) < tmp_dist:
                        tmp_index = j
                        tmp_dist = np.linalg.norm(coord - np.array([x, y, z]))

                x, y, z = sphere_vertices[tmp_index]

                tmp_dist = sys.maxsize
                tmp_index = 0
                for j, coord in enumerate(cortex_vertices):
                    if np.linalg.norm(coord - np.array([x, y, z])) < tmp_dist:
                        tmp_index = j
                        tmp_dist = np.linalg.norm(coord - np.array([x, y, z]))

                coord_to_plot.append(cortex_vertices[tmp_index])
            coord_to_plot = np.array(coord_to_plot)
            current_color = [self.all_color[i] if i != 0 and i != -1 else ("Crimson" if i == 0 else "amber") for i in label]
            scatters.append(
                go.Scatter3d(
                    x=coord_to_plot[:, 0],
                    y=coord_to_plot[:, 1],
                    z=coord_to_plot[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=current_color, opacity=0.8)
                )
            )

        vertices = cortex_mesh.darrays[0].data.astype(float)
        faces = cortex_mesh.darrays[1].data.astype(int)

        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='gray',
            opacity=0.50
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=True,
            title=self.title
        )

        scatters.append(mesh)

        fig = go.Figure(data=scatters, layout=layout)

        fig.update_layout(scene=dict(aspectmode='data'))

        fig.show()

    def plot_graphs(
            self,
            folder_path: str,
            radius=100
    ) -> None:

        for graph in os.listdir(folder_path):
            graph = get_graph_from_pickle(os.path.join(folder_path, graph))
            coord = []
            label = []
            for i in range(len(graph.nodes)):
                if len(graph.nodes[i]) != 0:
                    coord.append(graph.nodes[i]["coord"])
                    label.append(graph.nodes[i]["label"])

            current_color = [self.all_color[i] if i != 0 and i != -1 else ("Crimson" if i == 0 else "amber") for i in label]

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

        self.fig.show()
