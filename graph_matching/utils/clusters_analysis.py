"""This module contains code to analyse clusters generated

..moduleauthor:: Marius Thorre, Rohit Yadav
"""

from graph_matching.utils import graph_processing as gp
import numpy as np
import networkx as nx
import pickle as p
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


def insert_at(
        arr,
        output_size,
        indices
):
    result = np.zeros(output_size, dtype=np.uint16)
    existing_indices = [np.setdiff1d(np.arange(axis_size), axis_indices, assume_unique=True)
                        for axis_size, axis_indices in zip(output_size, indices)]
    result[np.ix_(*existing_indices)] = arr
    return result


def get_assignment_from_labelling(
        list_graphs: list,
        labelling_attribute_name,
        excluded_labels: bool = False
):
    """ Compute the assignment matrix based on the labeling stored in the nodes of the graphs in list_graphs,
        as the node attribute labelling_attribute_name

    :param list_graphs: list of graphs to work on
    :param labelling_attribute_name: node attribute used to store labeling for which we compute the assignment matrix
    :param excluded_labels:
    :return: assign_mat the assignment matrix and unique_labels the set of labels found across all graphs and ordered
        according to the rows of the computed assign_mat
    """

    all_graphs_labels = get_labelling_from_attribute(list_graphs, labelling_attribute_name)
    list_all_graphs_labels = concatenate_labels(all_graphs_labels)
    unique_labels = list(set(list_all_graphs_labels))
    if excluded_labels:
        print('excluded labels ', excluded_labels)
        for ex_lab in excluded_labels:
            unique_labels.remove(ex_lab)
    unique_labels.sort()
    tot_nb_nodes = len(list_all_graphs_labels)
    print('total number of nodes across all graphs', tot_nb_nodes)
    print('number of different labels stored in graphs', len(unique_labels))
    print('labels stored in graphs', unique_labels)
    # relabelling to get continuous labels that will correspond to row of the assignment matrix
    row_index_list_all_graphs_labels = list()
    for i in list_all_graphs_labels:
        if i in unique_labels:  # handle excluded labels: these are not relabeled
            idx = unique_labels.index(i)  # take the index of i in the set of labels
            row_index_list_all_graphs_labels.append(idx)  # making the relabeled list
    print('row index corresponding to labels', set(row_index_list_all_graphs_labels))

    assign_semimat = np.zeros((tot_nb_nodes, len(unique_labels)), dtype=np.uint16)
    if excluded_labels is not None:
        for ind_node, label in zip(range(tot_nb_nodes), row_index_list_all_graphs_labels):
            if label not in excluded_labels:
                assign_semimat[ind_node, label] = 1
    else:
        for ind_node, label in zip(range(tot_nb_nodes), row_index_list_all_graphs_labels):
                assign_semimat[ind_node, label] = 1

    X = assign_semimat @ assign_semimat.T

    sizes_dummy = [nx.number_of_nodes(g) for g in list_graphs]
    dummy_mask = [list(nx.get_node_attributes(graph, 'is_dummy').values()) for graph in list_graphs]
    dummy_mask = sum(dummy_mask, [])
    dummy_indexes = [i for i in range(len(dummy_mask)) if dummy_mask[i] == True]

    X_w_dummy = insert_at(X, (sum(sizes_dummy), sum(sizes_dummy)),
                               (dummy_indexes, dummy_indexes))

    return X, X_w_dummy, unique_labels


def concatenate_labels(all_labels):
    cat_labels = list()
    for l in all_labels:
        cat_labels.extend(l)
    return cat_labels


def nb_labelled_nodes_per_label(u_labs, all_labels):
    u_l_count = list()
    for u_l in u_labs:
        subj_u = list()
        for subj_labs in all_labels:
            subj_u.append(np.sum(subj_labs == u_l))
        u_l_count.append(subj_u)
    return np.array(u_l_count)


def get_labelling_from_attribute(list_graphs, labelling_attribute_name):
    all_labels = list()
    for g in list_graphs:
        labels = list(nx.get_node_attributes(g, labelling_attribute_name).values())
        all_labels.append(labels)

    return all_labels


