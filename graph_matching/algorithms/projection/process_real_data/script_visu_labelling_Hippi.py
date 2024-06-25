import sys
import os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import numpy as np
import networkx as nx
import pickle as p

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)

    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))

    nodes_attributes = {}
    for ind, node in enumerate(graph_no_dummy.nodes):
        nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}

    nx.set_node_attributes(graph_no_dummy, nodes_attributes)
    return one_nodes_coords_scaled

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/_obsolete_OASIS_full_batch/modified_graphs')
    path_to_match_mat = os.path.join(project_path, 'data/_obsolete_OASIS_full_batch/Hippi_res_real_mat.npy')
    path_to_r_perm = os.path.join(project_path, 'data/r_perm.gpickle')

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    Hippi = np.load(path_to_match_mat)
    X = [Hippi]

    nb_graphs = 134
    mesh = sio.load_mesh(template_mesh)

    largest_ind = 24
    g_l = p.load(open(os.path.join(project_path, "data/_obsolete_OASIS_full_batch/modified_graphs/graph_" + str(largest_ind) + ".gpickle"), "rb"))
    color_label_ordered = label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm = p.load(open(path_to_r_perm, "rb"))
    color_label = color_label_ordered[r_perm]
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)

    default_value = -0.1
    nb_nodes = len(g_l.nodes)

    for j in range(len(list_graphs)):
        grph = list_graphs[j]
        nodes_to_remove = gp.remove_dummy_nodes(grph)
        nodes_to_remove = np.where(np.array(nodes_to_remove) == False)
        grph.remove_nodes_from(list(nodes_to_remove[0]))
        nb_nodes = len(grph.nodes)
        row_scope = range(j * nb_nodes, (j + 1) * nb_nodes)

        if len(grph.nodes) == 101:
            break

    for matching_matrix in X:
        last_index = 0
        nb_unmatched = 0
        for i in range(nb_graphs):
            g = p.load(open(os.path.join(project_path, "data/_obsolete_OASIS_full_batch/modified_graphs/graph_" + str(i) + ".gpickle"), "rb"))
            nodes_to_remove = gp.remove_dummy_nodes(g)
            nodes_to_remove = np.where(np.array(nodes_to_remove) == False)
            g.remove_nodes_from(list(nodes_to_remove[0]))
            nb_nodes = len(g.nodes)

            if i == 0:
                col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
                prev_nb_nodes = nb_nodes
                perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int)
                transfered_labels = np.ones(nb_nodes) * default_value
                last_index += nb_nodes
            else:
                col_scope = range(last_index, last_index + nb_nodes)
                last_index += nb_nodes
                perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int)
                transfered_labels = np.ones(nb_nodes) * default_value

            for node_indx, ind in enumerate(row_scope):
                match_index = np.where(perm_X[node_indx, :] == 1)[0]
                if len(match_index) > 0:
                    transfered_labels[match_index[0]] = color_label[node_indx]
            nb_unmatched += np.sum(transfered_labels == default_value)
            nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
            s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=transfered_labels, nodes_mask=None, c_map='nipy_spectral')
            vb_sc.add_to_subplot(s_obj)

        print('nb_unmatched', nb_unmatched)
        vb_sc.preview()
