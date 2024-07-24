"""This module contains tools to generate barycenter graph from graph list
..Moduleauthor:: Marius Thorre
"""
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph_matching.utils.display_graph_tools import Visualisation

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
from graph_matching.utils.graph_processing import get_graph_from_pickle

graph_test_path = os.path.join(project_path, "resources/graph_for_test/")
file_cortex_mesh = os.path.join(project_path, "resources", "template_mesh", "lh.OASIS_testGrp_average_inflated.gii")
file_sphere_mesh = os.path.join(project_path, "resources", "template_mesh", "ico100_7.gii")
folder_path = os.path.join(
    project_path,
    "resources",
    "graph_for_test",
    "generation",
    "without_outliers",
    "noise_01"
)

G0 = get_graph_from_pickle(
    os.path.join(
        project_path,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_60",
        "graph_00000.gpickle"
    )
)

G10 = get_graph_from_pickle(
    os.path.join(
        project_path,
        "resources",
        "graph_for_test",
        "generation",
        "without_outliers",
        "noise_60",
        "graph_00010.gpickle"
    )
)

graphs = [G10, G0]

b = Barycenter(
    graphs=graphs,
    nb_node=30
)
b._check_node()
b.compute()

bary = b.get_graph()


v = Visualisation(
    graph=bary,
    sphere_radius=100,
    title="1 without outliers"
)

matrix = np.array([
    [-4.91646969, -0.54851652, 4.54434237],
    [-10.18405843, 1.92196881, 6.58853852],
    [6.4805479, 0.81961775, -5.56260351],
    [0.13756362, -3.19200932, 3.87055186],
    [-7.8682953, 0.33250164, 4.14606508],
    [-5.55409131, 1.25387522, 3.06631384],
    [4.40880484, -2.87867044, 0.70299703],
    [13.38886287, 14.02569404, -0.12529959],
    [-0.53052059, -4.3744531, -2.02934664],
    [-6.41120535, 1.06707813, 5.82052916],
    [-7.50759978, -2.54618497, -0.34549987],
    [-10.44930496, -0.17578329, 3.21539728],
    [5.73419207, -1.34596729, 1.30384561],
    [-1.54883064, -4.24401179, 0.04759445],
    [-4.55924432, -3.10286834, 0.71239048],
    [-3.80790509, -3.678474, 0.02900693],
    [-5.86407314, -3.22622274, 0.45655629],
    [-5.6046138, -1.65172642, 2.17222765],
    [2.82461676, -1.53434086, -4.82301716],
    [6.4805479, 0.81961775, -5.56260351],
    [-4.62214376, -1.63800257, 6.4270179],
    [-3.77075816, -2.70214598, 8.77892069],
    [-7.78852147, -0.03154486, 3.16376017],
    [1.56230403, -3.39186297, 5.95243],
    [-7.20361728, -2.93539225, 0.86028976],
    [-9.8858482, -0.97056249, 2.41946888],
    [-5.6046138, -1.65172642, 2.17222765],
    [-0.73347997, -2.98287088, 2.86816434],
    [-4.43975191, -3.10212348, 2.32572798],
    [13.39277641, 8.03712728, -1.83208132]
])

g2 = nx.Graph()

for i, coord in enumerate(matrix):
    g2.add_node(i, coord=coord*10, label=i)

v.add_graph_to_plot(G0)
v.add_graph_to_plot(G10)
v.add_graph_to_plot(g2)
v.show_fig()



