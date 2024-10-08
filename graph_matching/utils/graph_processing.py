import os
import numpy as np
import networkx as nx
import pickle
import math


def _compute_distance(adj_s, adj_t):
    """Compute euclidian distance between A and B adjacency matrix.
    :param np.ndarray A: adjacency matrix from source_graph
    :param np.ndarray B: adjacency matrix from target_graph
    :return: distance matrix
    """
    dist = np.zeros((adj_s.shape[0], adj_t.shape[0]))
    for a, i in zip(adj_s, range(len(adj_s))):
        for b, j in zip(adj_t, range(len(adj_t))):
            dist[i, j] = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return dist


def get_distance_between_graphs(first_graph: nx.Graph, graphs: list):
    node_distances = {}
    for node_bary in range(len(first_graph.nodes)):
        node_coord = first_graph.nodes[node_bary]["coord"]
        node_label = first_graph.nodes[node_bary]["label"]
        distance = []
        for graph in graphs:
            for node_graph in range(len(graph.nodes)):
                if graph.nodes[node_graph]["label"] == node_label:
                    distance.append(np.linalg.norm(node_coord - graph.nodes[node_graph]["coord"]))
        node_distances[node_label] = np.mean(distance)
    return node_distances


def check_point_on_sphere(points: np.ndarray, radius: float) -> bool:
    """
    Check if a point is on the sphere.
    :param points: array of all sphere coordinates
    :param radius: radius of the sphere
    :return: if all points are on the sphere
    """
    distances = np.linalg.norm(points, axis=1)

    return np.allclose(distances, radius)


def get_graph_from_pickle(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_as_gpickle(path: str, graph: nx.Graph):
    with open(path + ".gpickle", "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def get_graph_coord(
        graph: nx.Graph,
        nb_dimension: int
) -> np.ndarray:
    graph_coord = np.zeros(shape=(nx.number_of_nodes(graph), nb_dimension))
    for node in graph.nodes(data=True):
        graph_coord[node[0]] = node[1]["coord"]

    return graph_coord


def list_to_dict(list_in):
    """
    converter used for pitsgraph to networkx conversion
    :param array:
    :return:
    """
    D = {}
    for i, l_i in enumerate(list_in):
        D[i] = l_i
    return D


def sphere_nearest_neighbor_interpolation(graph, sphere_mesh, coord_attribute='coord'):
    """
    For each node in the graph,
    find the closest vertex in the sphere mesh from the 'coord' attribute of each node
    :param graph:
    :param sphere_mesh:
    :return:
    """

    nodes_coords = graph_nodes_attribute(graph, coord_attribute)
    vertex_number1 = sphere_mesh.vertices.shape[0]

    #print('vert_template.shape', vert_template.shape[0])
    #print('vert_pits.shape', vert_pits.shape[0])
    nn = np.zeros(nodes_coords.shape[0], dtype=np.int64)
    for ind, v in enumerate(nodes_coords):
        #print(v)
        nn_tmp = np.argmin(np.sum(np.square(np.tile(v, (vertex_number1, 1)) - sphere_mesh.vertices), 1))
        nn[ind] = nn_tmp
    #print(nodes_coords.shape)
    #print(len(nn))
    #nx.set_node_attributes(graph, list_to_dict(nn), 'ico100_7_vertex_index_noreg')
    nx.set_node_attributes(graph, list_to_dict(nn), 'ico100_7_vertex_index')  # Non Registered Vertex

    return graph


def load_graphs_in_list(path_to_graphs_folder, suffix=".gpickle"):
    """
    Return a list of graph loaded from the path, ordered according to the filename on disk
    """
    g_files = []
    with os.scandir(path_to_graphs_folder) as files:
        for file in files:
            if file.name.endswith(suffix):
                g_files.append(file.name)

    g_files.sort()  # sort according to filenames

    list_graphs = [get_graph_from_pickle(os.path.join(path_to_graphs_folder, graph)) for graph in g_files]

    return list_graphs


def load_labelled_graphs_in_list(path_to_graphs_folder, hemi='lh'):
    """
    Return a list of graph loaded from the path
    """
    files = os.listdir(path_to_graphs_folder)
    files_to_load = list()
    for f in files:
        if '.gpickle' in f:
            if hemi in f:
                files_to_load.append(f)
    list_graphs = []
    for file_graph in files_to_load:
        path_graph = os.path.join(path_to_graphs_folder, file_graph)
        graph = get_graph_from_pickle(path_graph)
        list_graphs.append(graph)

    return list_graphs


def graph_nodes_to_coords(graph, index_attribute, mesh):
    vert_indices = list(nx.get_node_attributes(graph, index_attribute).values())
    coords = np.array(mesh.vertices[vert_indices, :])
    return coords


def add_nodes_attribute(graph, list_attribute, attribute_name):
    """
    Given a graph, add to each node the corresponding attribute
    """

    attribute_dict = {}
    for node in graph.nodes:
        attribute_dict[node] = {attribute_name: list_attribute[node]}
    nx.set_node_attributes(graph, attribute_dict)


def graph_nodes_attribute(graph, attribute):
    """
    get the 'attribute' node attribute from 'graph' as a numpy array
    :param graph: networkx graph object
    :param attribute: string, node attribute to be extracted
    :return: a numpy array where i'th element corresponds to the i'th node in the graph
    if 'attribute' is not a valid node attribute in graph, then the returned array is empty
    """
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def graph_edges_attribute(graph, attribute):
    """
    get the 'attribute' edge attribute from 'graph' as a numpy array
    :param graph: networkx graph object
    :param attribute: string, node attribute to be extracted
    :return: a numpy array where i'th element corresponds to the i'th edge in the graph
    if 'attribute' is not a valid attribute in graph, then the returned array is empty
    """
    att = list(nx.get_edge_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    data_mask = np.ones_like(is_dummy)
    if True in is_dummy:
        graph.remove_nodes_from(np.where(np.array(is_dummy) == True)[0])
        inds_dummy = np.where(np.array(is_dummy) == True)[0]
        data_mask[inds_dummy] = 0
    return data_mask


def get_geodesic_distance_sphere(coord_a, coord_b, radius):
    '''
    Return the geodesic distance of two 3D vectors on a sphere
    '''
    return radius * np.arccos(np.clip(np.dot(coord_a, coord_b) / np.power(radius, 2), -1, 1))


def add_geodesic_distance_on_edges(graph):
    """
    Compute the geodesic distance represented by each edge
    and add it as attribute in the graph
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for edge in graph.edges:
        geodesic_distance = get_geodesic_distance_sphere(graph.nodes[edge[0]]["sphere_3dcoords"],
                                                         graph.nodes[edge[1]]["sphere_3dcoords"],
                                                         radius=100)

        edges_attributes[edge] = {"geodesic_distance": geodesic_distance}

    nx.set_edge_attributes(graph, edges_attributes)


def add_id_on_edges(graph):
    """
    Add an Id information on edge (integer)
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for i, edge in enumerate(graph.edges):
        edges_attributes[edge] = {"id": i}

    nx.set_edge_attributes(graph, edges_attributes)


def add_dummy_nodes(graph, nb_node_to_reach):
    """
    Add a given number of dummy nodes to the graph
    """
    for _ in range(graph.number_of_nodes(), nb_node_to_reach):
        graph.add_node(graph.number_of_nodes(), is_dummy=True)


def transform_3dcoords_attribute_into_ndarray(graph):
    """
    Transform the node attribute sphere_3dcoord from a list to a ndarray
    """
    # initialise the dict for atttributes on edges
    nodes_attributes = {}

    # Fill the dictionnary with the nd_array attribute
    for node in graph.nodes:
        nodes_attributes[node] = {"sphere_3dcoords": np.array(graph.nodes[node]["sphere_3dcoords"])}

    nx.set_node_attributes(graph, nodes_attributes)


def preprocess_graph(graph):
    """
    preprocessing of graphs
    :param graph:
    :return:
    """

    # transform the 3d attributes into ndarray
    transform_3dcoords_attribute_into_ndarray(graph)

    # Compute the geodesic distance for each node and add the id information
    add_geodesic_distance_on_edges(graph)

    # add ID identifier on edges
    add_id_on_edges(graph)

    # add the 'is_dummy' attribute to nodes, that will be used when manipulating dummy nodes later
    nx.set_node_attributes(graph, values=False, name="is_dummy")



