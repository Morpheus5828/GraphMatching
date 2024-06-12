"""This module contains scripts to create benchmark on digits sklearn dataset using Sinkhorn and SNS
..moduleauthor:: Marius Thorre
"""

import sys

import numpy as np
import OT_method_comparaison as OTmc
import graph_matching.algorithms.pairwise.kergm as kergm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.model_selection import cross_val_score


class OTBenchmark:
    def __init__(
            self,
            class_one: int,
            class_two: int,
            OT_algo: str,
            dataset,
            add_dimension: bool,
            alpha: float,
            display_perf: bool = False
    ):
        """Benchmark between optimal transport algorithm on dataset using KNN
        :param class_one: first target class
        :param class_two: second target class
        :param OT_algo: "sns" or "sinkhorn"
        :param dataset: dataset for binary classification
        :param add_dimension: add positional dimension
        :param alpha: regularisation parameter for OT algorithm, must be positive
        """
        self.display_perf = display_perf
        self.class_one = class_one
        self.class_two = class_two
        self.algo = OT_algo
        self.dataset = dataset
        self.add_dimension = add_dimension
        self.alpha = alpha
        self.sinkhorn_iterations = []

    def wassernstein_distance(
            self,
            source: np.ndarray,
            destination: np.ndarray
    ):
        """
        Compute Wassernstein distance:
        https://en.wikipedia.org/wiki/Wasserstein_metric
        :param source: source distribution
        :param destination: destination distribution
        :return: compute cost matrix between source and destination distribs
        """
        mu_s = np.ones((source.shape[0], 1)) / source.shape[0]
        mu_t = np.ones((source.shape[0], 1)) / source.shape[0]

        if self.add_dimension:
            source = self.add_position_feature(source)
            destination = self.add_position_feature(destination)
            cost = []
            for i in source:
                for j in destination:
                    tmp = 0
                    tmp += 0.5 * np.abs(i[0] - j[0])
                    tmp += 0.25 * np.abs(i[1] - j[1])
                    tmp += 0.25 * np.abs(i[2] - j[2])
                    cost.append(tmp)

            cost = np.array(cost).reshape((source.shape[0], destination.shape[0]))
        else:
            source = source.reshape((-1, 1))/255
            destination = destination.reshape((1, -1))/255

            cost = np.abs((source - destination)/source.shape[0])
        return self.run_ot_method(cost, mu_s, mu_t)

    def run_ot_method(
            self,
            cost: np.ndarray,
            mu_s: np.ndarray,
            mu_t: np.ndarray
    ) -> float:
        """Compute distance between two matrix using optimal transport
        :param cost: matrix which contains source and destination matrix features
        :param mu_s: vector with same probability
        :param mu_t: vector with same probability
        :return: distance
        """
        iteration = 0
        transport = None
        if self.algo == "sns":
            transport, iteration = kergm.sinkhorn_newton_sparse(
                cost=cost,
                mu_s=mu_s,
                mu_t=mu_t,
                rho=15,
                eta=self.alpha,
                N1=20,
                N2=20,
                tolerance=0.01
            )
        elif self.algo == "sinkhorn":
            transport, iteration = kergm.sinkhorn_method(
                x=cost,
                mu_s=np.squeeze(mu_s),
                mu_t=np.squeeze(mu_t),
                gamma=1 / self.alpha,
                tolerance=0.01)
        else:
            print("Method not recognized")

        self.sinkhorn_iterations.append(iteration)

        return (transport * cost).sum()

    def add_position_feature(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        :param X: initial matrix to add positional dimension
        :return: initial matrix but after added positional dimension
        """
        result = []
        c = 0
        n = int(np.sqrt(X.shape[0]))
        for i in range(n):
            for j in range(n):
                result.append([[X[c] / 255], [i / 7], [j / 7]])
                c += 1
        return np.array(result)

    def best_parameter(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ) -> float:
        """
        Compute best K-Nearest-Neighbors parameter using cross validation
        :param X_train:
        :param y_train:
        :return:
        """
        scores = []
        k_values = np.arange(1, 10)
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=self.wassernstein_distance)
            score = cross_val_score(knn, X_train, y_train, cv=5)
            scores.append(np.mean(score))
        return np.argmax(scores) + 1

    def get_perf(self, k=None):
        X_data, y_data = self.dataset
        X_data = X_data[np.logical_or(y_data == self.class_one, y_data == self.class_two)]
        y_data = y_data[np.logical_or(y_data == self.class_one, y_data == self.class_two)]

        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=0.3,
            random_state=42,
            stratify=y_data
        )
        if k is None:
            k = self.best_parameter(X_train, y_train)

        knn = KNeighborsClassifier(n_neighbors=k, metric=self.wassernstein_distance)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        if self.display_perf:
            print(f"best_k: {k}")
            print(classification_report(y_test, y_pred, target_names=[str(self.class_one), str(self.class_two)]))
            print(f"Meaning iteration: {np.mean(self.sinkhorn_iterations)}")

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    regularization = [1, 5, 10, 50, 100]
    """
    Display Iteration by Regularization
    """
    # OTmc.display_iteration_by_regularization_parameter(
    #     algo_one="sns",
    #     algo_two="sinkhorn",
    #     regularization=regularization,
    #     dataset=load_digits(return_X_y=True),
    #     class_one=0,
    #     class_two=1,
    #     add_dimension=True
    # )
    #
    # OTmc.display_iteration_by_regularization_parameter(
    #     algo_one="sns",
    #     algo_two="sinkhorn",
    #     regularization=regularization,
    #     dataset=load_digits(return_X_y=True),
    #     class_one=6,
    #     class_two=9,
    #     add_dimension=True
    # )

    """
    Display Accuracy by Regularization
    """
    # OTmc.display_accuracy_by_regularization_parameter(
    #     algo_one="sns",
    #     algo_two="sinkhorn",
    #     regularization=regularization,
    #     dataset=load_digits(return_X_y=True),
    #     class_one=0,
    #     class_two=1,
    #     add_dimension=True,
    #     k_parameter=3
    # )

    OTmc.display_accuracy_by_regularization_parameter(
        algo_one="sns",
        algo_two="sinkhorn",
        regularization=regularization,
        dataset=load_digits(return_X_y=True),
        class_one=0,
        class_two=1,
        add_dimension=True,
        k_parameter=2
    )




