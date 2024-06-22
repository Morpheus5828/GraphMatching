import os
import sys
import numpy as np
import pickle
import graph_matching.utils.graph_processing as gp
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    path_to_consistency = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/consistency')
    path_to_figs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/figures')

    reg_or_unreg = ''  # '_unreg'#''
    methods = ['MatchEig', 'media', 'media_no_excl', 'media_no_excl_neg_values', 'neuroimage', 'kerGM', 'mALS', 'mSync',
               'CAO']

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    nb_bins = 20
    bins = np.arange(0, 1, 0.05)
    dens = False
    fig1, ax = plt.subplots(1, len(methods), sharey=True, sharex=True)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)

        pickle_path = os.path.join(path_to_consistency, "nodeCstPerGraph_" + method + reg_or_unreg + ".pck")

        if not os.path.exists(pickle_path):
            print(f"Le fichier {pickle_path} n'existe pas.")
            continue

        with open(pickle_path, "rb") as pickle_in:
            nodeCstPerGraph = pickle.load(pickle_in)

        print(
            f"Average across all nodes of node consistency {method}{reg_or_unreg}: {np.mean(nodeCstPerGraph)}, {np.std(nodeCstPerGraph)}")

        ax[ind].hist(nodeCstPerGraph.flatten(), density=dens, bins=bins)  # density=False would make counts
        ax[ind].set_ylabel('Frequency')
        ax[ind].set_xlabel('Consistency')
        ax[ind].set_title(method)
        ax[ind].grid(True)

    plt.tight_layout()
    plt.show()
