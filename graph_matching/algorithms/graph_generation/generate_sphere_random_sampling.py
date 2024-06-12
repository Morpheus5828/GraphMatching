"""This module contains code to generate sphere random sampling
.. moduleauthor:: Marius Thorre, Rohit Yadav
"""

import numpy as np


def run(
    vertex_number: int = 100,
    radius: float = 1.0
) -> np.ndarray:
    """ Generate a sphere with random sampling
    :param vertex_number:
    :param radius:
    :return : sphere coordinate array
    :rtype : np.ndarray
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return coords