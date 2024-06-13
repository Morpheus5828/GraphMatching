import sys
import os
import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca
import numpy as np
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')  # lh.OASIS_testGrp_average_inflated.gii'
    label_attribute = 'label_neuroimage'  # 'label_media'

    path_to_graphs = os.path.join(project_path, 'data/labelled_pits_graphs_coords')
    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    print(len(list_graphs))

    for g in list_graphs:
        nx.set_node_attributes(g, values=False, name="is_dummy")

    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    labels = list(cluster_dict.keys())
    labels.sort()
    print(labels)
    print(len(labels))

    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

    vb_sc = gv.visbrain_plot(reg_mesh)
    vmin = 0
    vmax = 92  # 329#92

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        labels_subj = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels_subj])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None, c_map='nipy_spectral', vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                    nodes_size=60, nodes_mask=None, c_map='nipy_spectral', symbol='disc',
                                                    vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()

    vb_sc2 = gv.visbrain_plot(reg_mesh)
    label_to_plot = 2  # 222

    for ind, g in enumerate(list_graphs):
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        color_label_to_plot = np.ones(color_label.shape)
        color_label_to_plot[color_label == label_to_plot] = 0

        if np.sum(color_label == label_to_plot) == 0:
            print(ind)
        else:
            s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_to_plot, nodes_mask=None, c_map='nipy_spectral', vmin=0, vmax=1)
            vb_sc2.add_to_subplot(s_obj)

    vb_sc2.preview()
