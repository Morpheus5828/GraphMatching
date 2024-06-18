import os
import sys
import resources.slam.io as sio
import numpy as np
import networkx as nx
import scipy.io as sco
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    c_map = 'nipy_spectral'
    path_to_X = os.path.join(project_path, 'data/Oasis_original_new_with_dummy')
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_consistency = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/consistency')
    reg_or_unreg = ''  # '_unreg'
    method = 'media_no_excl'  # 'neuroimage'
    default_label = -0.1
    vmin = -0.1
    vmax = 1.1

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    print('----------------------------')
    print(method)
    largest_ind = 22
    label_attribute = 'labelling_from_assgn'

    # Load the assignment matrix
    if method in ['kmeans', 'neuroimage', 'media']:
        file_X = os.path.join(path_to_X, f"X_{method}{reg_or_unreg}_dummy.mat")
    else:
        file_X = os.path.join(path_to_X, f"X_{method}{reg_or_unreg}.mat")

    if method in ['kerGM', 'kmeans', 'neuroimage', 'media']:
        X = sco.loadmat(file_X)["full_assignment_mat"]
    else:
        X = sco.loadmat(file_X)['X']

    trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute,
                                                default_label_value=default_label)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=f'label_{method}')

    print(cluster_dict.keys())
    print(len(cluster_dict))

    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    for g in list_graphs:
        gp.remove_dummy_nodes(g)

    vb_sc = gv.visbrain_plot(reg_mesh)  # None

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None,
                                                        c_map=c_map, vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh,
                                               attribute_vertex_index='ico100_7_vertex_index')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                    nodes_size=90, nodes_mask=None, c_map=c_map, symbol='disc',
                                                    vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()
