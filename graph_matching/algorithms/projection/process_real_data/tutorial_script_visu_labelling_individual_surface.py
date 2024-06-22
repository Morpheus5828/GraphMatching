import sys
import os
import resources.slam.io as sio
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


if __name__ == "__main__":
    file_mesh = os.path.join(project_path, 'data/Oasis_all_subjects_white/FS_OASIS/OAS1_0439/surf/lh.white.gii')
    file_basins = os.path.join(project_path, 'data/OASIS_all_subjects/OAS1_0439/dpfMap/alpha_0.03/an0_dn20_r1.5/alpha0.03_an0_dn20_r1.5_L_area50FilteredTexture.gii')
    path_to_labelled_graphs = os.path.join(project_path, 'data/Oasis_original_new_with_dummy/labelled_graphs/')

    list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)
    graph = list_graphs[56]  # corresponds to nth subject

    # Load the mesh
    mesh = sio.load_mesh(file_mesh)

    # Load the basins texture
    tex_basins = sio.load_texture(file_basins)

    # Remove dummy nodes from the graph
    gp.remove_dummy_nodes(graph)

    # Create a texture for matching labels
    matching_labels_tex = np.zeros_like(tex_basins.darray[0].copy())
    for n in graph.nodes:
        lab_matching_value = graph.nodes.data()[n]['labelling_mSync']
        basin_tex_label = graph.nodes.data()[n]['basin_label']
        matching_labels_tex[tex_basins.darray[0] == basin_tex_label] = lab_matching_value

    # Plot the mesh with the matching labels texture
    vb_sc2 = gv.visbrain_plot(mesh, tex=matching_labels_tex, cmap='jet',
                              caption='Visu on individual mesh with basins',
                              cblabel='basins colors')

    # Get the coordinates of the nodes
    nodes_coords = gp.graph_nodes_to_coords(graph, 'vertex_index', mesh)

    # Visualize the graph nodes on the mesh
    s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(graph, nodes_coords, node_color_attribute='labelling_mSync',
                                                 edge_color_attribute=None, nodes_mask=None, c_map='jet')
    vb_sc2.add_to_subplot(s_obj2)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc2)
    vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=200)

    # Show the plot on the screen
    vb_sc2.preview()
