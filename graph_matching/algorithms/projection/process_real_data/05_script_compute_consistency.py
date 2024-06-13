import os
import sys
import pickle
import scipy.io as sco
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

path_to_consistency = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/consistency')
path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
path_to_X = os.path.join(project_path, 'data/Oasis_original_new_with_dummy')

if not os.path.exists(path_to_consistency):
    print(f"Le répertoire {path_to_consistency} n'existe pas.")
    sys.exit(1)
if not os.path.exists(path_to_graphs):
    print(f"Le répertoire {path_to_graphs} n'existe pas.")
    sys.exit(1)
if not os.path.exists(path_to_X):
    print(f"Le répertoire {path_to_X} n'existe pas.")
    sys.exit(1)

list_graphs = gp.load_graphs_in_list(path_to_graphs)
nb_graphs = len(list_graphs)
print('nb graphs ', nb_graphs)

methods = ['MatchEig']  # ['media_no_excl']

for ind, method in enumerate(methods):
    print('----------------------------')
    print(method)

    if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
        file_X = os.path.join(path_to_X, "X_" + method + "_dummy.mat")
    else:
        file_X = os.path.join(path_to_X, "X_" + method + ".mat")


    if not os.path.exists(file_X):
        print(f"Le fichier {file_X} n'existe pas.")
        continue  # Passer à la méthode suivante

    if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
        X = sco.loadmat(file_X)["full_assignment_mat"]
    else:
        X = sco.loadmat(file_X)['X']


    nb_nodes = int(X.shape[0] / nb_graphs)
    print(nb_nodes)


    nodeCstPerGraph = gca.compute_node_consistency(X, nb_graphs, nb_nodes)
    pickle_out_path = os.path.join(path_to_consistency, "nodeCstPerGraph_" + method + ".pck")
    with open(pickle_out_path, "wb") as pickle_out:
        pickle.dump(nodeCstPerGraph, pickle_out)

    print(f"Consistance des nœuds pour la méthode {method} sauvegardée dans {pickle_out_path}")

# Code de débogage supplémentaire (commenté pour l'instant)
# x_mSync = sco.loadmat(os.path.join(path_to_match_mat, "X_mSync.mat"))["X"]
# x_mALS = sco.loadmat(os.path.join(path_to_match_mat, "X_mALS.mat"))["X"]
# x_cao = sco.loadmat(os.path.join(path_to_match_mat, "X_cao_cst_o.mat"))["X"]
# x_kerGM = sco.loadmat(os.path.join(path_to_match_mat, "X_pairwise_kergm.mat"))["full_assignment_mat"]

# # Exemple pour mALS
# nb_nodes = int(x_mALS.shape[0] / nb_graphs)
# print(nb_nodes)
# nodeCstPerGraph_mALS = gca.compute_node_consistency(x_mALS, nb_graphs, nb_nodes)
# pickle_out_path_mALS = os.path.join(path_to_consistency, "nodeCstPerGraph_mALS.pck")
# with open(pickle_out_path_mALS, "wb") as pickle_out:
#     pickle.dump(nodeCstPerGraph_mALS, pickle_out)

# # Exemple pour mSync
# nb_nodes = int(x_mSync.shape[0] / nb_graphs)
# print(nb_nodes)
# nodeCstPerGraph_mSync = gca.compute_node_consistency(x_mSync, nb_graphs, nb_nodes)
# pickle_out_path_mSync = os.path.join(path_to_consistency, "nodeCstPerGraph_mSync.pck")
# with open(pickle_out_path_mSync, "wb") as pickle_out:
#     pickle.dump(nodeCstPerGraph_mSync, pickle_out)

# # Exemple pour kerGM
# nb_nodes = int(x_kerGM.shape[0] / nb_graphs)
# nodeCstPerGraph_KerGM = gca.compute_node_consistency(x_kerGM, nb_graphs, nb_nodes)
# pickle_out_path_kerGM = os.path.join(path_to_consistency, "nodeCstPerGraph_KerGM.pck")
# with open(pickle_out_path_kerGM, "wb") as pickle_out:
#     pickle.dump(nodeCstPerGraph_KerGM, pickle_out)

# # Exemple pour CAO
# nb_nodes = int(x_cao.shape[0] / nb_graphs)
# nodeCstPerGraph_CAO = gca.compute_node_consistency(x_cao, nb_graphs, nb_nodes)
# pickle_out_path_CAO = os.path.join(path_to_consistency, "nodeCstPerGraph_CAO.pck")
# with open(pickle_out_path_CAO, "wb") as pickle_out:
#     pickle.dump(nodeCstPerGraph_CAO, pickle_out)
