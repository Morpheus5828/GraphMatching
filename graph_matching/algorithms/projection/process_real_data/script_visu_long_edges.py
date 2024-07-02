import sys, os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    file_template_mesh = os.path.join(project_path, 'data/template_mesh/lh.OASIS_testGrp_average_inflated.gii')
    file_sphere_mesh = os.path.join(project_path, 'data/template_mesh/ico100_7.gii')
    path_to_graphs = os.path.join(project_path, 'data/_obsolete_OASIS_full_batch/modified_graphs')
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    geo_length_threshold = 80

    print('nb graphs to show : ', len(list_graphs))
    # load and reorient the template_mesh mesh
    template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
    # plot the template_mesh mesh
    vb_sc = gv.visbrain_plot(template_mesh, caption='Visu on template_mesh mesh of edges longer than ' + str(geo_length_threshold))

    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        nodes_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', template_mesh)
        # manage nodes
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(graph, nodes_coords)
        # manage edges
        c_obj = gv.graph_edges_select(graph, nodes_coords,
                                      edge_attribute='geodesic_distance',
                                      attribute_threshold=geo_length_threshold)

        vb_sc.add_to_subplot(s_obj)
        vb_sc.add_to_subplot(c_obj)

    vb_sc.preview()

    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    vb_sc2 = gv.visbrain_plot(sphere_mesh, caption='Visu on sphere of edges longer than ' + str(geo_length_threshold))

    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        nodes_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', sphere_mesh)
        # manage nodes
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(graph, nodes_coords)
        # manage edges
        c_obj = gv.graph_edges_select(graph, nodes_coords,
                                      edge_attribute='geodesic_distance',
                                      attribute_threshold=geo_length_threshold)

        vb_sc2.add_to_subplot(s_obj)
        vb_sc2.add_to_subplot(c_obj)

    vb_sc2.preview()
