import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import graph_matching.utils.graph_processing as gp
import graph_matching.utils.clusters_analysis as gca

def create_perm_from_labels(labels):
    U = np.zeros((len(labels), len(set(labels))))

    for node, label in zip(range(U.shape[0]), labels):
        U[node, label] = 1

    return U @ U.T

if __name__ == "__main__":
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_match_mat = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/')
    path_to_X = os.path.join(project_path, 'data/Oasis_original_new_with_dummy')

    if not os.path.exists(path_to_graphs):
        print(f"Le répertoire {path_to_graphs} n'existe pas.")
        sys.exit(1)
    if not os.path.exists(path_to_match_mat):
        print(f"Le répertoire {path_to_match_mat} n'existe pas.")
        sys.exit(1)
    if not os.path.exists(path_to_X):
        print(f"Le répertoire {path_to_X} n'existe pas.")
        sys.exit(1)

    # Charger les graphes
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    method = 'media'  # 'neuroimage', 'CAO', 'kerGM', 'mSync', 'mALS'
    excluded_labels = None  # [-2]

    # Obtenir les assignations des étiquettes
    X_met, X_met_w_dummy, labels_met = gca.get_assignment_from_labelling(list_graphs, labelling_attribute_name='label_'+method, excluded_labels=excluded_labels)
    print(X_met.shape)
    print(X_met_w_dummy.shape)

    X_dict = {
        'full_assignment_mat': X_met,
        'corresp_labels_rows': labels_met
    }

    X_w_dummy_dict = {
        'full_assignment_mat': X_met_w_dummy
    }

    # Sauvegarder les matrices d'assignation (décommenter si nécessaire)
    # import scipy.io as sco
    # sco.savemat(os.path.join(path_to_X, "X_"+method+"_no_excl_dummy.mat"), X_w_dummy_dict, do_compression='True')

    # Code de débogage supplémentaire (décommenter si nécessaire)
    # debug labels media
    # X_buggy = sco.loadmat(os.path.join(path_to_X, "X_" + method + "_buggy.mat"))['full_assignment_mat']
    # print(X_buggy.shape)
    # X_diff = X_buggy - X_met
    # print(np.max(X_diff))

    # labels_media = []
    # for g in list_graphs:
    #     labels_media.extend(list(nx.get_node_attributes(g, 'label_media').values()))
    # keys_media = {i: labels_media[i] for i in range(len(set(labels_media)))}
    # keys_media = list(keys_media.keys())
    # set_media = list(set(labels_media))
    # set_media.sort()
    # print(np.max(np.array(set_media) - np.array(labels_met)))

    # relabeled_media = []
    # for i in labels_media:
    #     if i in set_media:
    #         idx = set_media.index(i)
    #         relabeled_media.append(keys_media[idx])
    #     else:
    #         print(i)
    # X_media = create_perm_from_labels(relabeled_media)  # labels_media)

    # X_diff1 = X_media - X_met
    # X_diff2 = X - X_media
    # print(np.max(X_diff1))
    # print(np.max(X_diff2))
    # for i, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
