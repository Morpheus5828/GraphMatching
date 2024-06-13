import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import resources.slam.io as sio
import graph_matching.utils.graph.graph_visu as gv
import graph_matching.utils.graph.graph_processing as gp

if __name__ == "__main__":

    template_mesh_path = os.path.join(project_path,
                                      'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/labelled_pits_graphs_coords')
    r_perm_path = os.path.join(project_path, 'data/r_perm.gpickle')

    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    mesh = sio.load_mesh(template_mesh_path)

    largest_ind = 24
    g_l = list_graphs[largest_ind]


    reg_mesh = gv.reg_mesh(mesh)
    vb_sc1 = gv.visbrain_plot(reg_mesh)
    vb_sc2 = gv.visbrain_plot(reg_mesh)

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=None, nodes_mask=None,
                                                        c_map='nipy_spectral')
        vb_sc1.add_to_subplot(s_obj)

        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index_noreg', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=None, nodes_mask=None,
                                                        c_map='nipy_spectral', symbol='+')
        vb_sc2.add_to_subplot(s_obj)

    vb_sc1.preview()
    vb_sc2.preview()
