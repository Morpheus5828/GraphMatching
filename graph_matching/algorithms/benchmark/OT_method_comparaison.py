"""This module contains scripts to compare Sinkhorn and SNS on reel dataset
-> Digits https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
-> Fetch olivetti face https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
..moduleauthor:: Marius Thorre
"""

import numpy as np
import matplotlib.pyplot as plt
from benchmark import OTBenchmark
from tqdm.auto import tqdm


def display_iteration_by_regularization_parameter(
        algo_one: str,
        algo_two: str,
        regularization: list,
        dataset,
        class_one: int,
        class_two: int,
        add_dimension: bool
):
    """Display graph with Number of Iteration by Regularisation parameter
    :param algo_one: first OT algo method
    :param algo_two: second OT algo method
    :param regularization: array of regularisation parameter
    :param dataset: dataset for binary classification
    :param class_one: first dataset target class
    :param class_two: second dataset target class
    :param add_dimension: boolean for add positional feature
    :return: display graph
    """
    assert len(regularization) > 3
    sinkhorn_iteration = []
    sns_iteration = []
    for alpha in tqdm(regularization):
        #     benchmark = OTBenchmark(
        #         class_one=class_one,
        #         class_two=class_two,
        #         OT_algo=algo_one,
        #         dataset=dataset,
        #         add_dimension=add_dimension,
        #         alpha=alpha
        #     )
        #     benchmark.get_perf()
        #     sns_iteration.append(np.barycenter(benchmark.sinkhorn_iterations))
        benchmark = OTBenchmark(
            class_one=class_one,
            class_two=class_two,
            OT_algo=algo_two,
            dataset=dataset,
            add_dimension=add_dimension,
            alpha=alpha
        )
        benchmark.get_perf()
        sinkhorn_iteration.append(np.mean(benchmark.sinkhorn_iterations))

    plt.plot(regularization, sinkhorn_iteration, label="Sinkhorn", c="red")
    # plt.plot(regularization, sns_iteration, label="Sinkhron Newton Stage", c="blue")
    plt.xscale("log")
    plt.legend()
    plt.ylabel("Iteration")
    plt.xlabel("Regularization parameter")
    if add_dimension:
        plt.title(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two}, \n with adding pixel position"
        )
        plt.savefig(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two} with adding pixel position")
    else:
        plt.title(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two}, \n without pixel position"
        )
        plt.savefig(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two} without pixel position")
    #plt.show()


def display_accuracy_by_regularization_parameter(
        algo_one: str,
        algo_two: str,
        regularization: list,
        dataset,
        class_one: int,
        class_two: int,
        add_dimension: bool,
        k_parameter: int
):
    """ Display graph with Accuracy KNN score by Regularisation parameter
    :param algo_one: first OT algo method
    :param algo_two: second OT algo method
    :param regularization: array of regularisation parameter
    :param dataset: dataset for binary classification
    :param class_one: first dataset target class
    :param class_two: second dataset target class
    :param add_dimension: boolean for add positional feature
    :param k_parameter: nb of neighbors for KNN algorithm
    :return: display graph
    """
    sinkhorn_score = []
    sns_score = []
    for alpha in tqdm(regularization):
        # benchmark = OTBenchmark(
        #     class_one=class_one,
        #     class_two=class_two,
        #     OT_algo=algo_one,
        #     dataset=dataset,
        #     add_dimension=add_dimension,
        #     alpha=alpha
        # )
        # sns_score.append(benchmark.get_perf(k_parameter))
        benchmark = OTBenchmark(
            class_one=class_one,
            class_two=class_two,
            OT_algo=algo_two,
            dataset=dataset,
            add_dimension=add_dimension,
            alpha=alpha
        )
        sinkhorn_score.append(benchmark.get_perf(k_parameter))
    plt.clf()
    plt.plot(regularization, sinkhorn_score, label="Sinkhorn", c="red")
    #plt.plot(regularization, sns_score, label="Sinkhron Newton Stage", c="blue")
    plt.xscale("log")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Regularization parameter")
    if add_dimension:
        plt.title(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two}, \n with adding pixel position"
        )
        plt.savefig(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two} with adding pixel position")
    else:
        plt.title(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two}, \n without adding pixel position"
        )
        plt.savefig(
            f"OT methods analyse on Digits dataset with class {class_one} and {class_two} without adding pixel position")
