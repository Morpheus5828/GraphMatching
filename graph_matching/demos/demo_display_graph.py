"""Example of displaying graph in 3 dimensions, using HTML file
.. moduleauthor:: Marius Thorre
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

import graph_matching.utils.graph.display_graph_tools as display

graph = display.get_graph_from_pickle("../graph_generation/graph_generated/0/reference_0.gpickle")

window = display.Visualisation(graph, title="Graph reference 0")
window.display()