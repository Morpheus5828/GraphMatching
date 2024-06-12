"""This module contains code to generate reference graph
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import numpy as np
from resources.slam import topology as stop
from graph_matching.utils.graph.graph_tools import *
import graph_matching.algorithms.graph_generation.generate_sphere_random_sampling as generate_sphere_random_sampling
import graph_matching.utils.graph.graph_processing as graph_processing


def run(
        nb_vertices: int,
        radius: float
) -> nx.Graph:
    """ Generate random sampling
    :param int nb_vertices:
    :param float radius:
    :return nx.Graph :
    :rtype: np.ndarray
    """
    sphere_random_sampling = generate_sphere_random_sampling.run(vertex_number=nb_vertices, radius=radius)
    sphere_random_sampling = tri_from_hull(sphere_random_sampling)  # Computing convex hull (adding edges)

    adja = stop.adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_array(adja.todense())
    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(sphere_random_sampling.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)

    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in graph.edges:
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = graph_processing.get_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)
    return graph

