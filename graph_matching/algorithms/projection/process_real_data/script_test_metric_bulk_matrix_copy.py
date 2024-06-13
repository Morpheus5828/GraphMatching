import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graph_matching.utils.graph.graph_processing as gp
from visbrain.objects import SourceObj, ColorbarObj

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)
    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords)) / (
            np.max(one_nodes_coords) - np.min(one_nodes_coords))

    nodes_attributes = {node: {"label_color": one_nodes_coords_scaled[ind]} for ind, node in enumerate(graph_no_dummy.nodes)}
    nx.set_node_attributes(graph_no_dummy, nodes_attributes)


def show_graph_nodes(graph, mesh, data, clim=(0, 1), transl=None):
    s_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', mesh)
    transl_bary = np.mean(s_coords)
    s_coords = 1.01 * (s_coords - transl_bary) + transl_bary

    if transl is not None:
        s_coords += transl

    s_obj = SourceObj('nodes', s_coords, color='red', edge_color='black', symbol='disc', edge_width=2.,
                      radius_min=30., radius_max=30., alpha=.9)
    s_obj.color_sources(data=data, cmap='hot', clim=clim)

    CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                      rect=(-.3, -2., 1., 4.), txtcolor='k')
    cb_obj = ColorbarObj(s_obj, cblabel='node consistency', border=False, **CBAR_STATE)

    return s_obj, cb_obj


if __name__ == "__main__":
    # Define paths
    path_to_graphs = os.path.join(project_path, "data/_obsolete_OASIS_full_batch/modified_graphs")
    Hippi_path = os.path.join(project_path, "data/RESULT_FRIOUL_HIPPI/Hippi_res_real_mat.npy")

    # Load graphs and data
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    Hippi = np.load(Hippi_path)

    nb_graphs = 134

    is_dummy_vect = []
    for g in list_graphs:
        is_dummy_vect.extend(list(nx.get_node_attributes(g, "is_dummy").values()))
    not_dummy_vect = np.logical_not(is_dummy_vect)

    # Calculate match percentages
    match_no_dummy_Hippi = 100 * np.sum(Hippi[:, not_dummy_vect], 1) / nb_graphs

    # Plot histogram
    nb_bins = 50
    plt.hist(match_no_dummy_Hippi, density=True, bins=nb_bins)
    plt.title('Distribution of Non-dummy Node Matches (Hippi)')
    plt.xlabel('Percentage of Matches')
    plt.ylabel('Frequency')
    plt.show()
