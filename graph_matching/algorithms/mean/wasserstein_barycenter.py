"""This module contains a computation of the Wasserstein barycenter
https://pythonot.github.io/auto_examples/gromov/plot_barycenter_fgw.html#

.. moduleauthor:: Marius THORRE
"""

import numpy as np
import networkx as nx
import ot.gromov
from scipy.sparse.csgraph import shortest_path
from matplotlib import cm
import matplotlib.colors as mcol
from ot.gromov._gw import fused_gromov_wasserstein
from ot.utils import list_to_array, unif, check_random_state, UndefinedParameter, dist
from ot.backend import get_backend
from ot.gromov._utils import update_feature_matrix, update_square_loss, update_kl_loss
import graph_matching.algorithms.pairwise.fgw as fgw


class Barycenter:
    def __init__(
            self,
            graphs,
            size_bary: int,
            find_tresh_inf: float,
            find_tresh_sup: float,
            find_tresh_step: int,
            graph_vmin: float,
            graph_vmax: float,
            graph_title: str = "Barycenter"
    ):
        self.graphs = graphs
        self.size_bary = size_bary
        self.graph_title = graph_title
        self.find_tresh_inf = find_tresh_inf
        self.find_tresh_sup = find_tresh_sup
        self.find_tresh_step = find_tresh_step
        self.graph_vmin = graph_vmin
        self.graph_vmax = graph_vmax
        self.C, self.A = self.compute()

    def find_thresh(self):
        dist = []
        search = np.linspace(self.find_tresh_inf, self.find_tresh_sup, self.find_tresh_step)
        for thresh in search:
            Cprime = self.sp_to_adjacency(0, thresh)
            SC = shortest_path(Cprime, method='D')
            SC[SC == float('inf')] = 100
            dist.append(np.linalg.norm(SC - self.C))
        return search[np.argmin(dist)], dist

    def sp_to_adjacency(self, threshinf=0.2, threshsup=1.8):
        H = np.zeros_like(self.C)
        np.fill_diagonal(H, np.diagonal(self.C))
        C = self.C - H
        C = np.minimum(np.maximum(C, threshinf), threshsup)
        C[C == threshsup] = 0
        C[C != 0] = 1
        return C

    def graph_colors(self, nx_graph):
        cnorm = mcol.Normalize(vmin=self.graph_vmin, vmax=self.graph_vmax)
        cpick = cm.ScalarMappable(norm=cnorm, cmap='viridis')
        cpick.set_array([])
        val_map = {}
        for k, v in nx.get_node_attributes(nx_graph, 'attr_name').items():
            val_map[k] = cpick.to_rgba(v)
        colors = []
        for node in nx_graph.nodes():
            colors.append(val_map[node])
        return colors

    def get_attributes(self):
        Ys = []
        Cs = []

        for graph in self.graphs:
            coord = []
            for n in range(len(graph.nodes)):
                coord.append(np.array(graph.nodes[n]["coord"]))
            Ys.append(coord)
            Cs.append(nx.adjacency_matrix(graph).todense())

        return Cs, Ys

    def compute(self):
        Cs, Ys = self.get_attributes()
        C, A = ot.gromov.fgw_barycenters(N=self.size_bary, Ys=Ys, Cs=Cs, alpha=0.5)
        #C, A = self.fgw_barycenters(self.size_bary, Ys, Cs, alpha=0.5)
        return C, A

    def fgw_barycenters(self,
                        N, Ys, Cs, ps=None, lambdas=None, alpha=0.5, fixed_structure=False,
                        fixed_features=False, p=None, loss_fun='square_loss', armijo=False,
                        symmetric=True, max_iter=100, tol=1e-9, stop_criterion='barycenter',
                        warmstartT=False, verbose=False, log=False, init_C=None, init_X=None,
                        random_state=None, **kwargs):

        if stop_criterion not in ['barycenter', 'loss']:
            raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

        Cs = list_to_array(*Cs)
        Ys = list_to_array(*Ys)
        arr = [*Cs, *Ys]
        if ps is not None:
            arr += list_to_array(*ps)
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

        d = Ys[0].shape[1]

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
            if warmstartT:
                res = [fused_gromov_wasserstein(
                    Ms[s], C, Cs[s], p, ps[s], loss_fun=loss_fun, alpha=alpha, armijo=armijo, symmetric=symmetric,
                    G0=T[s], max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                    for s in range(S)]
            else:
                res = [fgw.conditional_gradient(
                    C1=C,
                    C2=Cs[s],
                    mu_s=p,
                    mu_t=ps[s],
                    distance=Ms[s],
                    gamma=0.3,
                    eta=600,
                    rho=80,
                    N1=50,
                    N2=50,
                    ot_method="sinkhorn",
                    tolerance=1e-5
                ) for s in range(S)]

                # sinkhorn 0.3, tolerance 1e-5
                # res = [fused_gromov_wasserstein(
                #     Ms[s], C, Cs[s], p, ps[s], loss_fun=loss_fun, alpha=alpha, armijo=armijo, symmetric=symmetric,
                #     G0=None, max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                #     for s in range(S)]
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

    def get_graph(self):
        G = nx.from_numpy_array(self.A)
        tmp = nx.Graph()
        for node, i in enumerate(G.nodes):
            tmp.add_node(node, sphere_3dcoords=self.C[i], vertex_index=node)
        return tmp

