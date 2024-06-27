"""
..moduleauthor:: Marius Thorre
"""
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
from graph_matching.utils.graph_processing import get_graph_from_pickle

mesh_cortex_path = 'C:/Users/thorr/OneDrive/Bureau/Stage/lh.OASIS_testGrp_average_inflated(1).gii'
mesh_spherical_path = "C:/Users/thorr/PycharmProjects/GraphMatching/graph_matching/data/template_mesh/ico100_7.gii"

graph_test_path = os.path.join(project_path, "resources/graph_for_test")
graphs = []
for g in os.listdir(os.path.join(graph_test_path, "generation")):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, "generation", g)))

b = Barycenter(
    graphs=graphs,
    nb_node=30
)
b.compute(fixed_structure=True)
bary = b.get_graph()

v = Visualisation(title="barycenter_on_cortex", graph=bary)

# v.plot_graph_on_mesh(
#     cortext_mesh_path=mesh_cortex_path,
#     sphere_mesh_path=mesh_spherical_path
# )

v.plot_all_graph_on_mesh(
    graphs=graphs,
    cortext_mesh_path=mesh_cortex_path,
    sphere_mesh_path=mesh_spherical_path
)

#v.save_as_html("C:/Users/thorr/OneDrive/Bureau/Stage")


