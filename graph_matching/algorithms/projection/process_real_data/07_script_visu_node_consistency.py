import os
import sys
import numpy as np
import pickle
import scipy.io as sco

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    path_to_X = os.path.join(project_path, 'data/Oasis_original_new_with_dummy')
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_consistency = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/consistency')
    path_to_figs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/figures')

    # Vérifier l'existence des fichiers et répertoires
    for path in [template_mesh, path_to_X, path_to_graphs, path_to_consistency]:
        if not os.path.exists(path):
            print(f"Le chemin {path} n'existe pas.")
            sys.exit(1)

    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    c_map = 'Reds_r'
    reg_or_unreg = ''
    method = 'kerGM'
    default_label = -0.1
    vmin = 0
    vmax = 0.5

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    print('----------------------------')
    print(method)
    # Calculer l'étiquetage à partir de la matrice d'assignation si nécessaire
    if 'media' in method:
        label_attribute = 'label_media'
    elif 'neuroimage' in method:
        label_attribute = 'label_neuroimage'
    else:
        largest_ind = 22
        label_attribute = 'labelling_from_assgn'
        # Charger la matrice d'assignation
        file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
        if 'kerGM' in method:
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']
        trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute, default_label_value=default_label)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    print(cluster_dict.keys())
    print(len(cluster_dict))

    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    # Charger la consistance des nœuds
    pickle_in_path = os.path.join(path_to_consistency, "nodeCstPerGraph_" + method + reg_or_unreg + ".pck")
    if not os.path.exists(pickle_in_path):
        print(f"Le fichier {pickle_in_path} n'existe pas.")
        sys.exit(1)
    with open(pickle_in_path, "rb") as pickle_in:
        nodeCstPerGraph = pickle.load(pickle_in)

    clusters_cst = gca.get_consistency_per_cluster(cluster_dict, nodeCstPerGraph)
    print(np.min(clusters_cst), np.max(clusters_cst))
    print(np.mean(clusters_cst), np.std(clusters_cst))
    print(clusters_cst)

    # Données à afficher au niveau des nœuds du graphe
    data_node_cstr = np.mean(nodeCstPerGraph, 1)

    vb_sc = gv.visbrain_plot(reg_mesh)
    for ind_g, g in enumerate(list_graphs):
        data_mask = gp.remove_dummy_nodes(g)
        data_node_cstr = nodeCstPerGraph[:, ind_g]
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
                                                        nodes_size=None, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh, attribute_vertex_index='ico100_7_vertex_index')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clusters_cst,
                                                    nodes_size=90, nodes_mask=None, c_map=c_map, symbol='disc',
                                                    vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc)

    vb_sc.preview()
    # vb_sc.screenshot(os.path.join(path_to_figs, 'consistency_' + method + '.png'))
