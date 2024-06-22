import numpy as np


def get_top_left(P):
    return np.diag(np.diag(P @ np.ones((P.shape[1], P.shape[0]))))


def get_bottom_right(P):
    return np.diag(np.diag(P.T @ np.ones((P.shape[0], P.shape[1]))))


def fusion(P):
    tl = get_top_left(P)
    br = get_bottom_right(P)

    delta_second = np.zeros((
        tl.shape[1] + P.shape[1],
        tl.shape[1] + P.shape[1]
    ))

    delta_second[
    :tl.shape[0],
    :tl.shape[1]
    ] = tl

    delta_second[
    delta_second.shape[0] - br.shape[0]:,
    delta_second.shape[1] - br.shape[1]:
    ] = br

    delta_second[
    P.T.shape[1]:,
    : P.T.shape[1]
    ] = P.T

    delta_second[
    :P.shape[0],
    P.shape[0]:
    ] = P

    return delta_second
