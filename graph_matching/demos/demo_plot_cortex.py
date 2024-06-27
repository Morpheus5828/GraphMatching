
import graph_matching.utils.graph_visu as gv
import graph_matching.utils.graph_processing as gp
import resources.slam.io as sio
import resources.slamviz.plot as splt

if __name__ == "__main__":

    file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
    simus_run = 0
    # path_to_graphs = '../data/simu_graph/noise_70,outliers_0/'+str(simus_run)+'/graphs'

    path_to_graphs = '../data/simu_graph/NEW_SIMUS_JULY_11/0/noise_100,outliers_varied/graphs/'
    list_graphs = gp.load_graphs_in_order(path_to_graphs)

    print(len(list_graphs))

    # Get the meshes
    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        # print(len(graph.nodes))
        gp.sphere_nearest_neighbor_interpolation(graph, sphere_mesh)

    density_map = gv.nodes_density_map(list_graphs, mesh, nb_iter=3, dt=0.5)

    # plt.figure()
    # plt.hist(density_map, bins=50)
    # plt.show()

    visb_sc = splt.visbrain_plot(mesh=mesh,
                                 tex=density_map,
                                 caption='Template mesh',
                                 cblabel='density', cmap='jet')  # clim = cmap range, default = (min(data), max(data))

    visb_sc1 = splt.visbrain_plot(mesh=sphere_mesh,
                                  tex=density_map,
                                  caption='Sphere mesh',
                                  cblabel='density', cmap='jet')

    # visb_sc = splt.visbrain_plot(mesh=sphere_mesh,
    #                         tex=density_map,
    #                         caption='Sphere mesh',
    #                          cblabel='density', visb_sc=visb_sc)

    visb_sc.preview()
    visb_sc1.preview()