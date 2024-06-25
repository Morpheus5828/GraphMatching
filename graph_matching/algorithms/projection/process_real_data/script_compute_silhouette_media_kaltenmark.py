import os
import sys
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import graph_matching.utils.clusters_analysis as gca
import numpy as np
import scipy.io as sco
import pickle as p

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    # Define paths
    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/modified_graphs')
    template_mesh_path = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_silhouette = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/silhouette')
    reg_or_unreg = '_unreg'  # Modify this based on your requirement

    # Define paths to assignment matrices
    path_to_mALS = os.path.join(project_path, f"data/Oasis_original_new_with_dummy/X_mALS{reg_or_unreg}.mat")
    path_to_mSync = os.path.join(project_path, f"data/Oasis_original_new_with_dummy/X_mSync{reg_or_unreg}.mat")
    path_to_CAO = os.path.join(project_path, "data/Oasis_original_new_with_dummy/X_CAO.mat")
    path_to_kerGM = os.path.join(project_path, f"data/Oasis_original_new_with_dummy/X_pairwise_kergm{reg_or_unreg}.mat")

    # Load mesh
    mesh = sio.load_mesh(template_mesh_path)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    # Remove dummy nodes from graphs
    for i, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)

    # Load assignment matrices
    X_mALS = sco.loadmat(path_to_mALS)['X']
    X_mSync = sco.loadmat(path_to_mSync)['X']
    X_CAO = sco.loadmat(path_to_CAO)['X']
    X_kerGM = sco.loadmat(path_to_kerGM)["full_assignment_mat"]

    largest_ind = 22  # Adjust based on your largest graph index
    label_attribute = f'labelling_kerGM{reg_or_unreg}'

    # Get labelling from assignment
    gca.get_labelling_from_assignment(list_graphs, X_kerGM, largest_ind, mesh, label_attribute)

    # Create clusters lists
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)

    # Calculate centroids
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

    # Calculate silhouette values
    silhouette_dict = gca.get_all_silhouette_value(list_graphs, cluster_dict)

    # Save silhouette values
    silhouette_path = os.path.join(path_to_silhouette, f'{label_attribute}_silhouette.gpickle')
    with open(silhouette_path, "wb") as pickle_out:
        p.dump(silhouette_dict, pickle_out)

    # Load silhouette values (for debugging or verification purposes)
    # with open(silhouette_path, "rb") as pickle_in:
    #     silhouette_dict = p.load(pickle_in)

    clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)

    # Visualize the results
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)
    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                    nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                    vmin=-1, vmax=1)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()

    print(f'Mean silhouette: {np.mean(clust_silhouette)}')
    print(f'STD silhouette: {np.std(clust_silhouette)}')
    print(f'Number of clusters: {len(clust_silhouette)}')
