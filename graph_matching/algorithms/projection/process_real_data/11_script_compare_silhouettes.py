import os
import sys
import pickle as p
import matplotlib.pyplot as plt
import graph_matching.utils.clusters_analysis as gca
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    # template_mesh = os.path.join(project_path, 'data/template_mesh/ico100_7.gii')
    template_mesh = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new/')
    path_to_silhouette = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/silhouette')
    methods = ['MatchEig', 'CAO', 'kerGM', 'mALS', 'mSync', 'media', 'neuroimage']  # ,'kmeans_70_real_data_dummy','kmeans_90_real_data_dummy','kmeans_110_real_data_dummy']

    nb_bins = 20
    dens = False

    fig1, ax = plt.subplots(2, len(methods), sharey=True, sharex=False)

    clust_silhouettes = list()
    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)
        pickle_out = open(os.path.join(path_to_silhouette, 'labelling_' + method + '_silhouette.gpickle'), "rb")
        silhouette_dict = p.load(pickle_out)
        pickle_out.close()
        clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)
        nb_nodes = np.sum(clust_nb_nodes)
        clust_nb_nodes_perc = [100 * v / nb_nodes for v in clust_nb_nodes]
        print(np.mean(clust_silhouette))
        print(np.std(clust_silhouette))
        print(len(clust_silhouette))
        sort_ind = np.argsort(clust_silhouette)
        print(np.array(list(silhouette_dict.keys()))[sort_ind])
        print(np.array(clust_silhouette)[sort_ind])
        print(clust_nb_nodes_perc)
        if -0.1 in silhouette_dict.keys():
            ind_c = list(silhouette_dict.keys()).index(-0.1)
            print(clust_nb_nodes_perc[ind_c])
        clust_silhouettes.append(clust_silhouette)

        ax[0, ind].hist(clust_silhouette, density=dens, bins=nb_bins)  # density=False would make counts
        ax[0, ind].set_ylabel('Frequency')
        ax[0, ind].set_xlabel('silhouette')
        ax[0, ind].set_title(method)
        ax[0, ind].grid(True)

        ax[1, ind].hist(clust_nb_nodes_perc, density=dens, bins=nb_bins)  # density=False would make counts
        ax[1, ind].set_ylabel('Frequency')
        ax[1, ind].set_xlabel('perc of tot nodes')
        ax[1, ind].set_title(method)
        ax[1, ind].grid(True)

    plt.show()
