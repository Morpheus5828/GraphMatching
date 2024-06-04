from graph_matching.utils.graph.graph_tools import *
from scipy.stats import betabinom

def _generate_nb_outliers_and_nb_supress(
        nb_vertices: int
) -> tuple:
    """ Sample nb_outliers and nb_supress from a Normal distance following the std of real data
    :param nb_vertices:
    :return: Tuple which contains nb outliers and nb supress
    :rtype: (int, int)
    """
    # mean_real_data = 40         # mean real data
    std_real_data = 4  # std real data

    mu = 10  # mu_A = mu_B = mu
    sigma = std_real_data
    n = 25

    alpha = compute_alpha(n, mu, sigma ** 2)  # corresponding alpha with respect to given mu and sigma
    beta = compute_beta(alpha, n, mu)  # corresponding beta

    nb_supress = betabinom.rvs(n, alpha, beta, size=1)[0]
    nb_outliers = betabinom.rvs(n, alpha, beta, size=1)[0]  # Sample nb_outliers

    return int(nb_outliers), int(nb_supress)