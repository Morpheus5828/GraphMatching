import numpy as np


def _get_gradient(c_C1_C2, C1, C2, distance, transport, alpha=0.5, q=2.0):
    """Compute Gradient using eq (7)
    :param np.ndarray c_C1_C2: constant from eq (6)
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param np.ndarray transport: vector which contains current transportation map
    :param float alpha: fixed to 1/2, it's the equilibrum between cost and distances
    :param float q: fixed to 2.0, it's L2 Loss parametter
    :return: gradient
    """
    return (1 - alpha) * (distance ** q) + 2.0 * alpha * c_C1_C2 - C1 @ transport @ (2.0 * C2).T


def _get_constant(
        C1: np.ndarray,
        C2: np.ndarray,
        distance: np.ndarray,
        transport: np.ndarray,
        alpha: float = 0.5
) -> np.ndarray:
    """Compute constant from eq (6) in Gromov-Wasserstein Averaging of Kernel and Distance Matrices
    by Peyré.G, Cuturi.M, Solomon.J
    :param np.ndarray C1: cost matrix of the source graph
    :param np.ndarray C2: cost matrix of the target graph
    :param np.ndarray distance: euclidian distance matrix between both graphs
    :param np.ndarray transport: vector which contains current transportation map
    :param alpha:
    :return: float
    """
    transport = transport.reshape((-1, 1))
    result = transport.flatten().T @ (-2 * alpha * np.kron(C2, C1)) @ transport.flatten()
    result += (1 - alpha * distance).T.flatten() @ transport.flatten()
    return np.argmin(result)