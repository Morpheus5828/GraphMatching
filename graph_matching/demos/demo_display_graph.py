"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import os
import sys
import webbrowser
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.utils.graph_processing import get_graph_from_pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Extract gpickle graph file
pickle_path = os.path.join(script_dir, "graph_generated", "pickle", "noise_100_outliers_varied/graph_00000.gpickle")
#pickle_path = os.path.join(script_dir, "graph_generated", "pickle", "reference.gpickle")
graph = get_graph_from_pickle(pickle_path)
for node in range(len(graph.nodes)):
    print(graph.nodes[node])
window = Visualisation(graph, title="reference", sphere_radius=100)
window.construct_sphere()

graph_path = os.path.join(script_dir, "graph_generated", "html")
window.save_as_html(graph_path)
# print(os.path.join(graph_path, window.title +".html"))
webbrowser.open(os.path.join(graph_path, window.title +".html"))

