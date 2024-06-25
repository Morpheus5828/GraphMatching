import sys
import os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import graph_matching.utils.clusters_analysis as gca
import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt

from graph_matching.utils.display_graph_tools import Visualisation


def save_labelled_graphs(list_graphs, path_to_save):
    for i in range(len(list_graphs)):
        Visualisation(list_graphs[i]).save_as_pickle(os.path.join(path_to_save, "graph_{:05d}".format(i)))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
    if project_path not in sys.path:
        sys.path.append(project_path)

    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_save_labelled_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/labelled_graphs')

    methods = ['mALS', 'mSync', 'CAO', 'MatchEig']

    trash_label = -2
    reg_or_unreg = ''
    largest_ind = 22
    default_label = -0.1
    nb_bins = 20
    dens = False

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    fig1, ax = plt.subplots(2, len(methods), sharey=True, sharex=False)

    for ind, method in enumerate(methods):
        path_to_X = os.path.join(project_path, "data/Oasis_original_new_with_dummy/X_" + method + reg_or_unreg + ".mat")
        print('----------------------------')
        print(method)

        if 'media' in method:
            label_attribute = 'label_media'
        elif 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        else:
            if ('kerGM' in method) or ('kmeans' in method):
                X = sco.loadmat(path_to_X)["full_assignment_mat"]
            else:
                X = sco.loadmat(path_to_X)['X']
            print(X.shape)
            print('get_clusters_from_assignment')
            label_attribute = 'labelling_' + method + reg_or_unreg
            gca.get_labelling_from_assignment(list_graphs, X, largest_ind, mesh, label_attribute, default_label_value=default_label)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)

        if 'media' in method:
            label_attribute = 'label_media'
        elif 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        else:
            label_attribute = 'labelling_' + method + reg_or_unreg

        transfered_labels_all_graphs = gca.get_labelling_from_attribute(list_graphs,
                                                                        labelling_attribute_name=label_attribute)

        a_transfered_labels_all_graphs = np.array(gca.concatenate_labels(transfered_labels_all_graphs))
        print(np.unique(a_transfered_labels_all_graphs))
        ax[0, ind].hist(a_transfered_labels_all_graphs, density=dens, bins=nb_bins)  # density=False would make counts
        ax[0, ind].set_ylabel('Frequency')
        ax[0, ind].set_xlabel('labels')
        ax[0, ind].set_title(method)
        ax[0, ind].grid(True)

    plt.show()

    vb_sc = gv.visbrain_plot(mesh)
    simbs = ['cross', 'ring', 'disc', 'square']
    for ind, method in enumerate(methods):
        if 'media' in method:
            label_attribute = 'label_media'
        elif 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        else:
            label_attribute = 'labelling_' + method + reg_or_unreg

        print(label_attribute)
        cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
        # Calculate the centroid
        centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")
        centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, mesh, attribute_vertex_index='ico100_7_vertex_index')
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=ind * np.ones(centroids_3Dpos.shape[0],),
                                                        nodes_size=15, nodes_mask=None, c_map='jet', symbol=simbs[ind],
                                                        vmin=0, vmax=len(methods))

        vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()
