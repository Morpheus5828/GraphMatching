"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

import graph_matching.utils.graph.display_graph_tools as display

graph = display.get_graph_from_pickle("graph_matching/demos/graph_generated/0/reference_0.gpickle")

window = display.Visualisation(graph, title="Graph reference 0")
window.display()