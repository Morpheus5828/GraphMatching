import pickle

import networkx as nx
import numpy as np
import os
import sys
import graph_matching.utils.clusters_analysis as gca
import pickle as p

from graph_matching.utils.display_graph_tools import Visualisation

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

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

def transform_3d_coordinates_into_ndarray(graph):
    """
    Transform the node attribute sphere_3dcoord from a list to a ndarray
    """
    # initialise the dict for atttributes on edges
    nodes_attributes = {}

    # Fill the dictionnary with the nd_array attribute
    for node in graph.nodes:
        nodes_attributes[node] = {"sphere_3dcoords": np.array(graph.nodes[node]["sphere_3dcoords"])}

    nx.set_node_attributes(graph, nodes_attributes)

def dummy_for_visu(g, max_size):
    len_g = len(g.nodes)
    num_nodes_add = max_size - len_g

    dummy_dict = {}
    for i in range(nx.number_of_nodes(g)):
        dummy_dict[i] = {'is_dummy': False}

    nx.set_node_attributes(g, dummy_dict)

    if num_nodes_add > 0.:
        for i in range(num_nodes_add):
            g.add_node(len_g + i, is_dummy=True)

if __name__ == "__main__":
    path_new_graphs = os.path.join(project_path, 'data/_obsolete_OASIS_labelled_pits_graphs/')
    path_to_save = os.path.join(project_path, 'data/check_graph_conversion')
    label_attribute = 'label_media'  # 'label_neuroimage'

    pick_corr = open(os.path.join(project_path, 'data/graph_correspondence_new.pickle'), "rb")
    corr_dict = p.load(pick_corr)
    pick_corr.close()

    original_names = [f[0] for f in corr_dict]
    new_graphs = []
    for graph in original_names:
        with open(os.path.join(path_new_graphs, graph), "rb") as f:
            new_graphs.append(pickle.load(f))
    print(len(new_graphs))
    for g in new_graphs:
        nx.set_node_attributes(g, values=False, name="is_dummy")
    cluster_dict = gca.create_clusters_lists(new_graphs, label_attribute=label_attribute)
    labels = list(cluster_dict.keys())
    labels.sort()
    print(labels)
    print(len(labels))

    all_graphs_name = os.listdir(path_new_graphs)

    all_graphs_nums = list()
    for i, graph in enumerate(new_graphs):
        graph_num = int(all_graphs_name[i].split('_')[1].split('.')[0])
        all_graphs_nums.append([all_graphs_name[i].split('.')[0], graph_num])
        Visualisation(graph=graph, title="graph_{:05d}".format(graph_num)).save_as_pickle(path_to_save)

    print(all_graphs_nums)
    print(len(all_graphs_nums))
