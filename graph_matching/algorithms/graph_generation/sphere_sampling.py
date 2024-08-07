"""This module contains tool to generate random point on a sphere
..moduleauthor:: Sylvain TAKERKART, Marius THORRE

"""

import numpy as np


def fibonacci(nb_point, radius):
    inc = np.pi * (3 - np.sqrt(5))
    off = 2. / nb_point
    k = np.arange(0, nb_point)
    y = k * off - 1. + 0.5 * off
    r = np.sqrt(1 - y * y)
    phi = k * inc
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    return x * radius, y * radius, z * radius
