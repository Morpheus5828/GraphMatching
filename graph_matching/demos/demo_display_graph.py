"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import os
import sys
from graph_matching.utils.display_graph_tools import Visualisation

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_cortex_mesh = os.path.join(project_root, "resources", "template_mesh", "lh.OASIS_testGrp_average_inflated.gii")
file_sphere_mesh = os.path.join(project_root, "resources", "template_mesh", "ico100_7.gii")
folder_path = os.path.join(project_root, "resources", "graph_for_test", "generation", "noise_10_outliers_varied")

window = Visualisation(title="noise_10", sphere_radius=100)

#window.plot_graphs(folder_path=folder_path, radius=100)

window.plot_all_graph_on_mesh(
    folder_path=folder_path,
    sphere_mesh_path=file_sphere_mesh,
    cortext_mesh_path=file_cortex_mesh,
)

