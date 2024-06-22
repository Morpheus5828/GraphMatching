import sys
import os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii')
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    path_to_graphs = os.path.join(project_path, 'data/Oasis_original_new/')
    file_sphere_mesh = os.path.join(project_path, 'data/template_mesh/ico100_7.gii')
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    mask_slice_coord = -15
    vb_sc = None
    inds_to_show = [0, 1, 11]
    graphs_to_show = [list_graphs[i] for i in inds_to_show]

    for g in graphs_to_show:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)

        if nodes_coords is None or len(nodes_coords) == 0:
            print(f"Erreur: Coordonnées des nœuds non valides pour le graphe {g}")
            continue

        s_obj, c_obj, node_cb_obj = gv.show_graph(g, nodes_coords, node_color_attribute=None, nodes_size=30,
                                                  c_map='nipy_spectral')

        if s_obj is None or c_obj is None:
            print(f"Erreur: Objet(s) Visbrain non valide(s) pour le graphe {g}")
            continue

        vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)

        vb_sc.add_to_subplot(c_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] - 1)
        vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] - 1)

    vb_sc.preview()

    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    vb_sc2 = gv.visbrain_plot(sphere_mesh)

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', sphere_mesh)

        if nodes_coords is None or len(nodes_coords) == 0:
            print(f"Erreur: Coordonnées des nœuds non valides pour le graphe {g}")
            continue

        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords)

        if s_obj is None:
            print(f"Erreur: Objet Visbrain non valide pour le graphe {g}")
            continue

        vb_sc2.add_to_subplot(s_obj)

    vb_sc2.preview()
