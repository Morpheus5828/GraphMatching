import sys, os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    template_mesh = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    path_to_graphs = os.path.join(project_path, 'data/_obsolete_OASIS_full_batch/modified_graphs')

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for g in list_graphs:
        print(len(g))

    # Get the mesh
    mesh = sio.load_mesh(template_mesh)
    vb_sc = gv.visbrain_plot(mesh)
    for ind_g, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        s_obj = gv.graph_nodes_coords_to_sources(g)
        vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()
