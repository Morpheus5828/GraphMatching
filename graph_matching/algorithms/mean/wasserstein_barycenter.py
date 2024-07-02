"""This module contains a computation of the Wasserstein barycenter
https://pythonot.github.io/auto_examples/gromov/plot_barycenter_fgw.html

.. moduleauthor:: Marius THORRE
"""

import numpy as np
import networkx as nx
import ot.gromov
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from matplotlib import cm
import matplotlib.colors as mcol
from geopy.distance import geodesic
from ot.gromov._gw import fused_gromov_wasserstein
from ot.utils import list_to_array, unif, check_random_state, UndefinedParameter, dist
from ot.backend import get_backend
from ot.gromov._utils import update_feature_matrix, update_square_loss, update_kl_loss
import graph_matching.algorithms.pairwise.fgw as fgw
from sklearn.neighbors import KNeighborsClassifier
from geopy.distance import geodesic


class Barycenter:
    def __init__(
            self,
            graphs: list,
            nb_node: int,
            title: str = "Barycenter"
    ):
        """
        Compute graphs barycenter
        :param graphs: list of graphs
        :param nb_node: barycenter node number that we want
        :param graph_title: graph title
        """
        self.graphs = graphs
        self.nb_node = nb_node
        self.graph_title = title
        self.F, self.A, self.T = None, None, None

    def compute(self, fixed_structure=False) -> None:
        adj_matrices = []
        nodes_positions = []
        nodes_label = []
        for graph in self.graphs:
            adj_matrices.append(nx.adjacency_matrix(graph).todense())
            nodes = []
            label = []
            for index in range(len(graph.nodes)):
                nodes.append(graph.nodes[index]["coord"])
                label.append(graph.nodes[index]["label"])
            nodes_positions.append(np.array(nodes))
            nodes_label.append(np.array(label))
        if fixed_structure:
            self.F, self.A = ot.gromov.fgw_barycenters(
                N=self.nb_node,
                Ys=nodes_positions,
                Cs=adj_matrices,
                alpha=0.5,
                fixed_structure=True,
                init_C=adj_matrices[0]
            )

        else:

            self.F, self.A, log = ot.gromov.fgw_barycenters(
                N=self.nb_node,
                Ys=nodes_positions,
                Cs=adj_matrices,
                alpha=0.05,
                log=True
            )

            self.T = log["T"][-1]

    def get_graph(self) -> nx.Graph:
        all_node_coord = []
        all_node_label = []

        for graph in self.graphs:
            for index in range(len(graph.nodes)):
                all_node_coord.append(graph.nodes[index]["coord"])
                all_node_label.append(graph.nodes[index]["label"])

        G = nx.from_numpy_array(self.A)
        tmp = nx.Graph()

        for node, i in enumerate(G.nodes):
            tmp.add_node(node, coord=self.F[i], label=np.argmax(self.T[i]))
        return tmp

    def get_label(self):
        pass

        # label = []
        # max_node = max([len(graph.nodes) for graph in self.graphs])
        # for i in range(max_node):
        #     label_i = np.array([graph.nodes[i]["label"] for graph in self.graphs])
        #     if 0 in label_i:
        #         label.append(0)
        #     elif -1 in label_i:
        #         occurrence = np.unique(label_i)
        #         if len(occurrence) > 3

    def _check_node(self):
        print()
        for graph in self.graphs:
            for node in range(len(graph.nodes)):
                print(graph.nodes[node])
            print()

        # max_node = max([len(graph.nodes) for graph in self.graphs])
        # for graph in self.graphs:
        #     nb_node_to_add = max_node - len(graph.nodes)
        #     for i in range(nb_node_to_add):
        #         graph.add_node(max(graph.nodes) + i + 1, coord=np.array([0, 0]), label=-1)
        # for graph in self.graphs:
        #     for node in range(len(graph.nodes)):
        #         print(graph.nodes[node])
        #     print()

    def get_distance_diff(self):
        dist = {}
        bary = self.get_graph()
        for node in range(len(bary)):
            current_dist = []
            for graph in self.graphs:
                for g_node in range(len(graph.nodes)):
                    if graph.nodes[g_node]["label"] == bary.nodes[node]["label"]:
                        current_dist.append(np.linalg.norm(graph.nodes[g_node]["coord"] - bary.nodes[node]["coord"]))
            dist[bary.nodes[node]["label"]] = np.mean(current_dist)
        print(dist)

    def fgw_barycenters(self,
                        N, Ys, Cs, ps=None, lambdas=None, alpha=0.5, fixed_structure=False,
                        fixed_features=False, p=None, loss_fun='square_loss', armijo=False,
                        symmetric=True, max_iter=100, tol=1e-9, stop_criterion='barycenter',
                        warmstartT=False, verbose=False, log=False, init_C=None, init_X=None,
                        random_state=None, **kwargs):

        arr = [*Cs, *Ys]
        if ps is not None:
            if isinstance(ps[0], list):
                raise ValueError(
                    "Deprecated feature in POT 0.9.4: weights ps[i] are lists and should be arrays from a supported backend (e.g numpy).")

            arr += [*ps]
        else:
            ps = [unif(C.shape[0], type_as=C) for C in Cs]
        if p is not None:
            arr.append(list_to_array(p))
        else:
            p = unif(N, type_as=Cs[0])

        nx = get_backend(*arr)

        S = len(Cs)
        if lambdas is None:
            lambdas = [1. / S] * S

        d = Ys[0].shape[1]  # dimension on the node features

        if fixed_structure:
            if init_C is None:
                raise UndefinedParameter('If C is fixed it must be initialized')
            else:
                C = init_C
        else:
            if init_C is None:
                generator = check_random_state(random_state)
                xalea = generator.randn(N, 2)
                C = dist(xalea, xalea)
                C = nx.from_numpy(C, type_as=ps[0])
            else:
                C = init_C

        if fixed_features:
            if init_X is None:
                raise UndefinedParameter('If X is fixed it must be initialized')
            else:
                X = init_X
        else:
            if init_X is None:
                X = nx.zeros((N, d), type_as=ps[0])

            else:
                X = init_X

        Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

        if warmstartT:
            T = [None] * S

        cpt = 0

        if stop_criterion == 'barycenter':
            inner_log = False
            err_feature = 1e15
            err_structure = 1e15
            err_rel_loss = 0.

        else:
            inner_log = True
            err_feature = 0.
            err_structure = 0.
            curr_loss = 1e15
            err_rel_loss = 1e15

        if log:
            log_ = {}
            if stop_criterion == 'barycenter':
                log_['err_feature'] = []
                log_['err_structure'] = []
                log_['Ts_iter'] = []
            else:
                log_['loss'] = []
                log_['err_rel_loss'] = []

        while ((err_feature > tol or err_structure > tol or err_rel_loss > tol) and cpt < max_iter):
            if stop_criterion == 'barycenter':
                Cprev = C
                Xprev = X
            else:
                prev_loss = curr_loss

            # get transport plans

            res = [fgw.conditional_gradient(
                distance=Ms[s],
                C1=C,
                C2=Cs[s],
                mu_s=p,
                mu_t=ps[s],
                ot_method="sns",
                eta=20,
                rho=70,
                N1=50,
                N2=50
            )
                for s in range(S)]
            # else:
            #     res = [fused_gromov_wasserstein(
            #         Ms[s], C, Cs[s], p, ps[s], loss_fun=loss_fun, alpha=alpha, armijo=armijo, symmetric=symmetric,
            #         G0=None, max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
            #         for s in range(S)]
            if stop_criterion == 'barycenter':
                T = res
            else:
                T = [output[0] for output in res]
                curr_loss = np.sum([output[1]['fgw_dist'] for output in res])

            # update barycenters
            if not fixed_features:
                Ys_temp = [y.T for y in Ys]
                X = update_feature_matrix(lambdas, Ys_temp, T, p, nx).T
                Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

            if not fixed_structure:
                if loss_fun == 'square_loss':
                    C = update_square_loss(p, lambdas, T, Cs, nx)

                elif loss_fun == 'kl_loss':
                    C = update_kl_loss(p, lambdas, T, Cs, nx)

            # update convergence criterion
            if stop_criterion == 'barycenter':
                err_feature, err_structure = 0., 0.
                if not fixed_features:
                    err_feature = nx.norm(X - Xprev)
                if not fixed_structure:
                    err_structure = nx.norm(C - Cprev)
                if log:
                    log_['err_feature'].append(err_feature)
                    log_['err_structure'].append(err_structure)
                    log_['Ts_iter'].append(T)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format(
                            'It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err_structure))
                    print('{:5d}|{:8e}|'.format(cpt, err_feature))
            else:
                err_rel_loss = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0. else np.nan
                if log:
                    log_['loss'].append(curr_loss)
                    log_['err_rel_loss'].append(err_rel_loss)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format(
                            'It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err_rel_loss))

            cpt += 1

        if log:
            log_['T'] = T
            log_['p'] = p
            log_['Ms'] = Ms

            return X, C, log_
        else:
            return X, C
