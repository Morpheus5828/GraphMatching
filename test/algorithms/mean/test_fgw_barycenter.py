import os
import sys
import networkx as nx
from unittest import TestCase
import numpy as np
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.algorithms.mean.fgw_barycenter import Barycenter
from graph_matching.utils.graph_processing import get_graph_from_pickle, get_distance_between_graphs

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


g1 = nx.Graph()
g1.add_node(0, coord=np.array([0, 0]), label=0)
g1.add_node(1, coord=np.array([0, 1]), label=2)
g1.add_node(2, coord=np.array([0, 2]), label=3)
g1.add_edge(0, 1)
g1.add_edge(1, 2)

g2 = nx.Graph()
g2.add_node(0, coord=np.array([1, 0]), label=0)
g2.add_node(1, coord=np.array([1, 1]), label=0)
g2.add_node(2, coord=np.array([1, 2]), label=2)
g2.add_edge(0, 1)
g2.add_edge(1, 2)

g3 = nx.Graph()
g3.add_node(0, coord=np.array([1, 0]), label=1)
g3.add_node(1, coord=np.array([1, 1]), label=2)
g3.add_node(2, coord=np.array([1, 2]), label=3)
g3.add_edge(0, 1)
g3.add_edge(1, 2)


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

graph_test_path = os.path.join(project_path, "resources/graph_for_test/generation/without_outliers/noise_60")
graphs = []
for g in os.listdir(graph_test_path):
    print(os.path.join(graph_test_path, g))

    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))


class TestWassersteinBarycenter(TestCase):
    def test_compute_g1_g2(self):
        b = Barycenter(
            graphs=[g1, g2],
            nb_node=3
        )

        b.compute(fixed_structure=True)
        truth_f = np.array([
            [0.5, 2.0],
            [0.5, 1],
            [0.5, 0]
        ])

        truth_a = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        self.assertTrue(np.array_equal(truth_f, b.F))
        self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute_g1_g2_g3(self):
        b = Barycenter(
            graphs=[g1, g2, g3],
            nb_node=3
        )

        b.compute(fixed_structure=True)

        truth_f = np.array([
            [0.66666667, 2.0],
            [0.66666667, 1.0],
            [0.66666667, 0.0]
        ])

        truth_a = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        self.assertTrue(np.allclose(truth_f, b.F))
        self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute_graph_test(self):
        b = Barycenter(
            graphs=[G0, G10],
            nb_node=30
        )

        b.compute()

        """
            barycenter value on graph with 3D coord nodes cannot have the same value
            juste visualize it with uncomment code after:
        """
        # v = Visualisation(title="barycenter", graph=b.get_graph(), sphere_radius=100)
        # v.construct_sphere()
        #
        # v.add_graph_to_plot(second_graph=G0)
        # v.add_graph_to_plot(second_graph=G10)
        # v.show_fig()

