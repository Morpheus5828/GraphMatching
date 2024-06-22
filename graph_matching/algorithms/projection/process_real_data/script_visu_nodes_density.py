import sys, os
import resources.slam.io as sio
import resources.slamviz.plot as splt
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":

    #path_to_graphs = '../data/OASIS_full_batch/modified_graphs'
    #path_to_graphs = '../data/simu_graph/' #simulated graphs
    #path_to_graphs = '../data/Oasis_original_new/'  # non registered graphs
    path_to_graphs = os.path.join(project_path, "data/HCP/modified_graphs_left")

    file_sphere_mesh = os.path.join(project_path, "data/lh.OASIS_testGrp_average_inflated.gii")
    print(file_sphere_mesh)
    sphere_mesh = sio.load_mesh(file_sphere_mesh)


    list_graphs = gp.load_graphs_in_list(path_to_graphs)


    graphs_nodes_number = list()
    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        graphs_nodes_number.append(len(graph.nodes))
        gp.sphere_nearest_neighbor_interpolation(graph, sphere_mesh)

    print('nodes numbers : ')
    print('mean=', np.mean(graphs_nodes_number))
    print('std=', np.std(graphs_nodes_number))
    print('sum across subjects=', np.sum(graphs_nodes_number))

    # using an average mesh as template
    template_mesh = '../data/template_mesh/ico100_7.gii'
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    density_map = gv.nodes_density_map(list_graphs, mesh, nb_iter=3, dt=0.5)

    plt.figure()
    plt.hist(density_map, bins=50)
    plt.show()

    # visb_sc = gv.visbrain_plot(mesh=mesh,
    #                         tex=density_map,
    #                         caption='density map',
    #                         cmap="jet",
    #                         #clim=(0, 0.03),
    #                         cblabel='mean curvature')

    visb_sc = splt.visbrain_plot(
        mesh=mesh, tex=density_map,
        caption='Template mesh',
        cblabel='density',
        cmap = 'jet'
    )

    #visb_sc.preview()

    # visb_sc2 = gv.visbrain_plot(mesh=sphere_mesh,
    #                         tex=density_map,
    #                         caption='density map',
    #                         cmap="jet")

    visb_sc = splt.visbrain_plot(
        mesh=sphere_mesh, tex=density_map,
        caption='Sphere mesh',
        cblabel='density',cmap ='jet', visb_sc=visb_sc
    )

    visb_sc.preview()





    # ## visualization of nodes as spheres
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
    #
    # # using a sphere mesh as template
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    # mesh = sio.load_mesh(template_mesh)
    #
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
