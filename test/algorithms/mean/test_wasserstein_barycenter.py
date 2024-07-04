import os
import sys
import networkx as nx
from unittest import TestCase
import numpy as np
from graph_matching.utils.display_graph_tools import Visualisation
from graph_matching.algorithms.mean.wasserstein_barycenter import Barycenter
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

g4 = nx.Graph()
g4.add_node(0, coord=np.array([1, 0]), label=1)
g4.add_node(1, coord=np.array([1, 1]), label=2)
g4.add_node(2, coord=np.array([1, 2]), label=3)
g4.add_node(3, coord=np.array([1, 3]), label=4)
g4.add_edge(0, 1)
g4.add_edge(1, 2)
g4.add_edge(2, 3)

g5 = get_graph_from_pickle(
    os.path.join(
        project_path,
        "resources/graph_for_test/generation/noise_1810/graph_00000.gpickle"
    )
)
g6 = get_graph_from_pickle(
    os.path.join(
        project_path,
        "resources/graph_for_test/generation/noise_1810/graph_00001.gpickle"
    )
)
gref = get_graph_from_pickle(
    os.path.join(
        project_path,
        "resources/graph_for_test/reference.gpickle"
    )
)

graph_test_path = os.path.join(project_path, "resources/graph_for_test/generation/noise_1810")
graphs = []
for g in os.listdir(graph_test_path):
    graphs.append(get_graph_from_pickle(os.path.join(graph_test_path, g)))


class TestWassersteinBarycenter(TestCase):

    def test_compute_g1_g2(self):
        b = Barycenter(
            graphs=[g1, g2],
            nb_node=3
        )

        b.compute(fixed_structure=True)
        print(b.A)
        # truth_f = np.array([
        #     [0.5, 2.0],
        #     [0.5, 1],
        #     [0.5, 0]
        # ])
        #
        # truth_a = np.array([
        #     [0, 1, 0],
        #     [1, 0, 1],
        #     [0, 1, 0]
        # ])
        #
        # self.assertTrue(np.array_equal(truth_f, b.F))
        # self.assertTrue(np.array_equal(truth_a, b.A))

    def test_compute_g1_g4(self):

        b = Barycenter(
            graphs=[g1, g4],
            nb_node=4
        )
        #print(nx.adjacency_matrix(g1))
        b._check_node()
        #print(b.get_label())


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

    def test_compute_g1_g2_g3_g4(self):
        b = Barycenter(
            graphs=[g1, g2, g3, g4],
            nb_node=3
        )

        b.compute(fixed_structure=True)

        truth_f = np.array([
            [0.75, 2.1875],
            [0.75, 1.125],
            [0.75, 0.0625]
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
            graphs=graphs,
            nb_node=30
        )

        b.compute()

        v = Visualisation(title="barycenter", graph=b.get_graph(), sphere_radius=100)
        v.construct_sphere()

        v.plot_graphs(folder_path=os.path.join(graph_test_path, "generation"), radius=100)

    def test_compare_graph_reference(self):
        # b = Barycenter(
        #     graphs=graphs,
        #     nb_node=30
        # )
        #
        # b.compute()
        #
        # v = Visualisation(title="bary_gref", graph=b.get_graph(), sphere_radius=90)
        # v.construct_sphere()
        #
        # v.add_graph_to_plot(second_graph=gref, radius=90)
        # v.save_as_html(graph_test_path)

        v = Visualisation(title="data_ref", graph=gref, sphere_radius=90)
        v.construct_sphere()
        v.plot_graphs(folder_path=os.path.join(graph_test_path, "generation"), radius=90)
        v.save_as_html(graph_test_path)

    def test_get_distance_between_all_graph(self):
        b = Barycenter(
            graphs=graphs,
            nb_node=30
        )

        b.compute(fixed_structure=True)
        bary = b.get_graph()

        distances = get_distance_between_graphs(first_graph=bary, graphs=graphs)

        for d in distances.values():
            self.assertTrue(d < 5.0)
