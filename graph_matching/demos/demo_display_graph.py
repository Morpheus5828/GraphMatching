"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import graph_matching.utils.display_graph_tools as display

pickle_path = os.path.join(script_dir, "graph_generated", "pickle", "reference.gpickle")

graph = display.get_graph_from_pickle(pickle_path)
window = display.Visualisation(graph, title="Graph reference 0", sphere_radius=100)
window.save_as_html(os.path.join(script_dir, "graph_generated"))

