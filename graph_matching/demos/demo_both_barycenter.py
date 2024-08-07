"""This module contains tools to generate barycenter graph from graph list
..Moduleauthor:: Marius Thorre
"""
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from graph_matching.algorithms.mean.fgw_barycenter import Barycenter
from graph_matching.utils.graph_processing import get_graph_from_pickle
from graph_matching.utils.display_graph_tools import Visualisation

graph_test_path = os.path.join(project_path, "resources/graph_for_test/generation/without_outliers/noise_60")
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
        "graph_00001.gpickle"
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


graphs = []
for g in os.listdir(graph_test_path):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))


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

matrix = np.array([[26.93289492, -96.13630055, 4.91149811],
                   [-34.80026537, -87.29389871, 34.0456812],
                   [-8.5975298, -93.54627341, -34.20540009],
                   [21.86906973, -85.76932445, 45.76347225],
                   [-67.07260394, -73.7476839, -7.32513325],
                   [66.10818982, -66.95537522, -33.62384286],
                   [-32.45724273, -55.54670932, 75.72743824],
                   [-37.40040871, -57.52206706, -72.22952319],
                   [89.29785135, -34.21048466, 25.23307578],
                   [-88.00644684, -22.77398652, 41.61164221],
                   [35.19502915, -18.0706948, -91.83062146],
                   [18.65442925, -15.15426554, 96.18488306],
                   [-67.20767433, -27.28797061, -67.97978487],
                   [98.69163807, -10.17510627, -9.19559419],
                   [-56.74938528, -24.13746348, 78.71232308],
                   [-15.1062245, -0.17980953, -98.85224979],
                   [76.22319537, -10.69849211, 62.01931564],
                   [-95.5885446, 27.9809365, -8.71300554],
                   [64.4488661, 18.97229396, -74.0559938],
                   [7.3329576, 40.18290174, 90.7101127],
                   [-47.81794916, 42.93188033, -76.61192741],
                   [86.53380966, 48.99896632, -10.45457533],
                   [-65.19311823, 51.00470492, 56.06525641],
                   [22.05032726, 65.34522745, -71.78368232],
                   [63.05923487, 53.96003263, 49.53278119],
                   [-73.42232612, 65.99559009, -15.87380749],
                   [53.39785905, 73.79117212, -41.26677842],
                   [-12.80102419, 81.42749976, 51.78089443],
                   [-9.49735627, 92.13615491, -37.52492973],
                   [29.7019706, 94.92252038, 8.25148584]])

g2 = nx.Graph()

for i, coord in enumerate(matrix):
    g2.add_node(i, coord=coord, label=i)

v.plot_graphs(graph_test_path)
v.show_fig()
