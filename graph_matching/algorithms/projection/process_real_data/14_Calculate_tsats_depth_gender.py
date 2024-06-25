import os, sys
import pickle as p
import numpy as np
import resources.slam.io as sio
import graph_matching.utils.graph_processing as gp
import scipy.stats as stats
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.clusters_analysis as gca
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

def create_clusters_lists_with_label_gender(list_graphs, gender_corresp, label_attribute="label_dbscan"):
    result_dict = {}
    label_depths = {}
    label_gender = {}

    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"]:
                label_cluster = graph.nodes[node][label_attribute]

                if label_cluster in result_dict:
                    depth_value = graph.nodes[node]['depth']
                    result_dict[label_cluster].append((i_graph, node))
                    label_depths[label_cluster].append(depth_value)
                    label_gender[label_cluster].append(gender_corresp[i_graph])
                else:
                    depth_value = graph.nodes[node]['depth']
                    result_dict[label_cluster] = [(i_graph, node)]
                    label_depths[label_cluster] = [depth_value]
                    label_gender[label_cluster] = [gender_corresp[i_graph]]

    return result_dict, label_depths, label_gender

def separate_groups_by_label(label_gender, label_depths):
    label_gen_sep = []
    for key in label_gender.keys():
        M = []
        F = []
        for i in range(len(label_gender[key])):
            if label_gender[key][i] == 'F':
                F.append(label_depths[key][i])
            else:
                M.append(label_depths[key][i])
        label_gen_sep.append([M, F])
    return label_gen_sep

def calculate_tstats_and_pvalues(corresp, method='labelling_mALS'):
    result_dict, label_depths, label_gender = create_clusters_lists_with_label_gender(labeled_graphs, corresp, method)
    label_gen_sep = separate_groups_by_label(label_gender, label_depths)
    t_stats = {}

    for key, lst in zip(label_gender.keys(), label_gen_sep):
        res = stats.ttest_ind(a=lst[0], b=lst[1], equal_var=True)
        t_stats[key] = [res[0], res[1]]

    return t_stats

def drop_nan_get_tstats(t_stat_dict, centroid_dict):
    drop_k = []
    for k in t_stat_dict.keys():
        if math.isnan(t_stat_dict[k][0]):
            drop_k.append(k)

    for k in drop_k:
        t_stat_dict.pop(k)
        centroid_dict.pop(k)

    return np.array(list(t_stat_dict.values())), centroid_dict

if __name__ == '__main__':
    path_to_labelled_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/labelled_graphs')
    path_ro_correspondence = os.path.join(project_path, 'data/gender_correspondence.pickle')

    labeled_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)
    gender_corresp = np.array(p.load(open(path_ro_correspondence, 'rb')))[:, 2]  # gender corresp list

    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    simbs = ['cross', 'ring', 'disc', 'square']
    methods = ['mALS', 'mSync', 'CAO', 'kerGM', 'MatchEig', 'media', 'neuroimage']

    for ind, method in enumerate(methods):
        vb_sc = gv.visbrain_plot(mesh)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)

        if 'media' in method:
            label_attribute = 'label_media'
        elif 'neuroimage' in method:
            label_attribute = 'label_neuroimage'
        else:
            label_attribute = 'labelling_' + method

        print(label_attribute)

        cluster_dict, label_depths, label_gender = create_clusters_lists_with_label_gender(labeled_graphs, gender_corresp, label_attribute=label_attribute)
        centroid_dict = gca.get_centroid_clusters(labeled_graphs, cluster_dict, coords_attribute="sphere_3dcoords")
        t_stats_method = calculate_tstats_and_pvalues(gender_corresp, label_attribute)
        t_stat, centroid_dict = drop_nan_get_tstats(t_stats_method, centroid_dict)  # drop nan value obtained for labels that contain only 1 group from t-stats.
        centroids_3Dpos = gca.get_centroids_coords(centroid_dict, labeled_graphs, mesh, attribute_vertex_index='ico100_7_vertex_index')

        plotted_val = np.zeros_like(t_stat[:, 0])
        inds_sup = t_stat[:, 0] > 1.9777  # 2.6127
        if sum(inds_sup) > 0:
            plotted_val[inds_sup] = 2
        inds_inf = t_stat[:, 0] < -1.9777  # 2.6127
        if sum(inds_inf) > 0:
            plotted_val[inds_inf] = -2

        inds_sup = t_stat[:, 0] > 2.6127
        if sum(inds_sup) > 0:
            plotted_val[inds_sup] = 4.5  # t_stat[inds_sup, 0]
        inds_inf = t_stat[:, 0] < -2.6127
        if sum(inds_inf) > 0:
            plotted_val[inds_inf] = -4.5  # t_stat[inds_inf, 0]

        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=plotted_val,
                                                        nodes_size=30, nodes_mask=None, c_map='bwr', vmin=-5, vmax=5)
        vb_sc.add_to_subplot(s_obj)
        vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=300)
        vb_sc.preview()
