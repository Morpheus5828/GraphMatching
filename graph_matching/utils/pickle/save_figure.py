"""This module contains code to save Graph into pickle file
.. moduleauthor:: Marius Thorre
"""

import pickle


def _as_gpickle(path: str, graph):
    with open(path, "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)