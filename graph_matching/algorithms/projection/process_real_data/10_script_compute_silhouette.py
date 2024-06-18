import os
import sys
import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca
import numpy as np
import scipy.io as sco
import pickle as p

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":

    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    path_to_silhouette = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/silhouette')
    reg_or_unreg = ''  # '_unreg'
    methods = ['MatchEig']  # ['media','neuroimage'], ['mALS','CAO']
    largest_ind = 22
    default_label = -0.1
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    for ind, method in enumerate(methods):
        path_to_X = os.path.join(project_path, "data/Oasis_original_new_with_dummy/X_" + method + reg_or_unreg + ".mat")
        print('----------------------------')
        print(method)
        if 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        elif 'media' in method:
            label_attribute = 'label_media'
        else:
            if ('kerGM' in method) or ('kmeans' in method):
                X = sco.loadmat(path_to_X)["full_assignment_mat"]
            else:
                X = sco.loadmat(path_to_X)['X']
            print(X.shape)
            print('get_clusters_from_assignment')
            label_attribute = 'labelling_' + method + reg_or_unreg
            trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute,
                                                        default_label_value=default_label)
            print(np.unique(trans_l))

    for i, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        # print(i,len(g)) # allow to identify the largest graph (22)

    for ind, method in enumerate(methods):
        if 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        elif 'media' in method:
            label_attribute = 'label_media'
        else:
            label_attribute = 'labelling_' + method + reg_or_unreg
        print('create_clusters_lists')
        cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
        # Calculate the centroid
        print('get_centroid_clusters')
        centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

        print('get_all_silhouette_value')
        silhouette_dict = gca.get_all_silhouette_value(list_graphs, cluster_dict)
        pickle_out = open(os.path.join(path_to_silhouette, 'labelling_' + method + '_silhouette.gpickle'), "wb")
        p.dump(silhouette_dict, pickle_out)
        pickle_out.close()
        clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)

        print('visu')

        vb_sc = gv.visbrain_plot(reg_mesh)
        centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                        vmin=-1, vmax=1)

        vb_sc.add_to_subplot(s_obj)
        # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
        #                           col=3, width_max=200)
        vb_sc.preview()
        # vb_sc.screenshot(os.path.join(path_to_silhouette, label_attribute+'.png'))
        print(np.mean(clust_silhouette))
        print(np.std(clust_silhouette))
        print(len(clust_silhouette))
