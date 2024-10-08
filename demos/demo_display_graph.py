"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from graph_matching.utils.display_graph_tools import Visualisation

file_cortex_mesh = os.path.join(project_root, "resources", "template_mesh", "lh.OASIS_testGrp_average_inflated.gii")
file_sphere_mesh = os.path.join(project_root, "resources", "template_mesh", "ico100_7.gii")
folder_path = os.path.join(project_root, "resources", "graph_for_test", "generation", "without_outliers", "noise_01")

# display a simple graph on a sphere mesh
v = Visualisation(title="noise_181", sphere_radius=100)

v.plot_graphs(folder_path=folder_path, radius=100)

v.show_fig()
