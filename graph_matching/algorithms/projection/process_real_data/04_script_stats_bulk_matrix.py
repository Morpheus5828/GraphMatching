import sys
import os
import pickle as p
import numpy as np
import networkx as nx
import scipy.io as sco
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp

if __name__ == "__main__":
    template_mesh_path = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_X = os.path.join(project_path, 'data/Oasis_original_new_with_dummy')

    if not os.path.exists(template_mesh_path):
        print(f"Le fichier {template_mesh_path} n'existe pas.")
        sys.exit(1)
    if not os.path.exists(path_to_graphs):
        print(f"Le répertoire {path_to_graphs} n'existe pas.")
        sys.exit(1)
    if not os.path.exists(path_to_X):
        print(f"Le répertoire {path_to_X} n'existe pas.")
        sys.exit(1)

    c_map = 'hot'
    vmin = 80
    vmax = 100

    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh_path))
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    methods = ['media', 'media_no_excl']
    nb_graphs = len(list_graphs)
    print('nb graphs ', nb_graphs)

    is_dummy_vect = []
    for g in list_graphs:
        is_dummy_vect.extend(list(nx.get_node_attributes(g, "is_dummy").values()))
    not_dummy_vect = np.logical_not(is_dummy_vect)
    print('nb nodes per graph (incl. dummy nodes ', len(g))
    print('total nb of nodes ', len(g) * nb_graphs)
    print(len(is_dummy_vect))
    print(len(not_dummy_vect))
    print('total nb of dummy nodes ', np.sum(is_dummy_vect))
    print('total nb of non-dummy nodes', np.sum(not_dummy_vect))

    nb_bins = 50
    dens = False
    fig1, ax = plt.subplots(2, len(methods), sharey=True)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)
        if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            file_X = os.path.join(path_to_X, "X_" + method + "_dummy.mat")
        else:
            file_X = os.path.join(path_to_X, "X_" + method + ".mat")

        if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']

        match_no_dummy = np.sum(X[:, not_dummy_vect], 1)
        match_dummy = np.sum(X[:, is_dummy_vect], 1)
        print(max(match_no_dummy))
        print(max(match_dummy))

        absc = range(len(match_no_dummy))
        ax[0, ind].bar(absc, np.sort(match_no_dummy))
        ax[0, ind].set_ylabel('percent of total nb of nodes per label')
        ax[0, ind].set_xlabel('Data')
        ax[0, ind].set_title('real match for ' + method)
        ax[1, ind].bar(absc, np.sort(match_dummy))
        ax[1, ind].set_ylabel('percent of total nb of nodes per label')
        ax[1, ind].set_xlabel('Data')
        ax[1, ind].set_title('dummy match for ' + method)

    plt.show()
