"""This module contains code to save Graph into pickle file
.. moduleauthor:: Marius Thorre
"""

import pickle
import networkx as nx

def _as_gpickle(path: str, graph: nx.Graph):
    with open(path + ".gpickle", "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


