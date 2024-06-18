import sys
import os
import numpy as np
import pickle as p

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp
import graph_matching.utils.graph.clusters_analysis as gca

def farthest_point_sampling(coords):
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    N = coords.shape[0]

    maxdist = 0
    bestpair = ()
    for i in range(N):
        for j in range(i + 1, N):
            if dist_mat[i, j] > maxdist:
                maxdist = dist_mat[i, j]
                bestpair = (i, j)

    P = list()
    P.append(bestpair[0])
    P.append(bestpair[1])

    while len(P) < N:
        maxdist = 0
        vbest = None
        for v in range(N):
            if v in P:
                continue
            for vprime in P:
                if dist_mat[v, vprime] > maxdist:
                    maxdist = dist_mat[v, vprime]
                    vbest = v
        P.append(vbest)

    return P

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, "data/template_mesh/lh.OASIS_testGrp_average_inflated.gii")
    path_to_graphs = os.path.join(project_path, "data/Oasis_original_new_with_dummy/modified_graphs")
    path_to_match_mat = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/')

    cmap = gv.rand_cmap(101, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    cmap = 'gist_ncar'

    ico_mesh = sio.load_mesh(os.path.join(project_path, 'data/template_mesh/ico100_7.gii'))
    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)

    largest_ind = 22
    g = list_graphs[largest_ind]
    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', ico_mesh)
    color_label_ordered = gca.label_nodes_according_to_coord(g, reg_mesh, coord_dim=0)
    color_label_ordered = color_label_ordered - 0.1
    r_perm = np.random.permutation(len(g))
    p.dump(r_perm, open(os.path.join(project_path, "data/r_perm_22.gpickle"), "wb"))
    color_label_r = color_label_ordered[r_perm]

    farthest_reordering = farthest_point_sampling(nodes_coords)
    color_label = color_label_ordered[farthest_reordering]

    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_ordered, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap, vmin=-0.1, vmax=1)
    vb_sc.add_to_subplot(s_obj)

    vb_sc1 = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap, vmin=-0.1, vmax=1)
    vb_sc1.add_to_subplot(s_obj)

    vb_sc2 = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_r, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap, vmin=-0.1, vmax=1)
    vb_sc2.add_to_subplot(s_obj)

    vb_sc.preview()
    vb_sc1.preview()
    vb_sc2.preview()
