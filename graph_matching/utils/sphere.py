"""This module contains tools to Plotting/sampling points on a sphere.
..moduleauthor:: Marius Thorre, Rohit Yadav
"""

import sys
sys.path.append("/graph_matching/utils")
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from graph_matching.utils.vMF import sample_vMF


class Sphere:
    def __init__(self):
        self.sphere = []
        self.samples = []
        self.x = None
        self.y = None
        self.z = None
        self.distribution = ['uniform', 'vMF']

    def make_sphere(self, radius: float) -> tuple:
        """
        Get mesh for a sphere of a given radius.
        :param radius: float value
        :return: tuple which contains 3 np.ndarray sphere coordinate
        :rtype: tuple
        """

        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
        self.x = radius * sin(phi) * cos(theta)
        self.y = radius * sin(phi) * sin(theta)
        self.z = radius * cos(phi)

        self.sphere = (self.x, self.y, self.z)
        return self.sphere

    def draw_sphere(self, ax, radius: float):
        """
        Draw an empty sphere
        :param ax: axis where to plot sphere
        :param radius: value of sphere radius
        """
        x, y, z = self.make_sphere(radius - 0.01)  # subtract a little so points show on sphere
        ax.plot_surface(
            x,
            y,
            z,
            rstride=1, cstride=1,
            color=sns.xkcd_rgb["light grey"],
            alpha=0.5,
            linewidth=0
        )

    def plot(self, radius: float = 1.0, data: tuple = None):
        """
        Plot a sphere with samples superposed, if supplied.

        :param radius: radius of the base sphere
        :param data: list of sample objects
        """

        sns.set_style('dark')
        ax = plt.figure().add_subplot(projection='3d')

        self.draw_sphere(ax=ax, radius=radius)

        if data == None:
            is_list = True
            data = self.samples
            N = len(data)
        else:
            is_list = False
            N = 1

        colors = [sns.color_palette("GnBu_d", N)[i] for i in reversed(range(N))]

        data_check = data[0] if is_list else  data

        if type(data_check) is vMFSample:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='$\kappa = $' + str(d.kappa), color=colors[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='$\kappa = $' + str(data.kappa), color=colors[i])


        elif type(data_check) is Coord3D:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='uniform samples', color=sns.color_palette("GnBu_d", N))
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='uniform samples', color=sns.color_palette("GnBu_d", N))


        else:
            print('Error: data type not recognised')

        ax.set_axis_off()
        ax.legend(bbox_to_anchor=[0.65, 0.75])

    def sample(
        self,
        nb_sample: int,
        radius: float = 1.0,
        distribution="uniform",
        mu = None,
        kappa = None
    ):
        """Sample points on a spherical surface.
        :param nb_sample: number of sample
        :param radius: float sphere radius
        :param distribution: type of distribution
        :param mu: parametter of vMF distrbution
        :param kappa: parametter of vMF distrbution
        """

        if distribution == 'uniform':
            u = np.random.uniform(0, 1, nb_sample)
            v = np.random.uniform(0, 1, nb_sample)

            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)

            # convert to cartesian
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi)
            self.samples = (x, y, z)

        elif distribution == 'vMF':
            try:
                s = sample_vMF(mu, kappa, nb_sample)
            except:
                print('Error: mu and kappa must be defined when sampling from vMF')
                return
            self.samples = vMFSample(s, kappa)

        else:
            print('Error: sampling distribution not recognised (try \'uniform\' or \'vMF\')')

        return self.samples


class vMFSample:
    """
    Store 3D coordinates from the sample_vMF function
    and kappa value.
    """

    def __init__(self, sample, kappa):
        """
        Store 3D coordinates from the sample_vMF function
        and kappa value.
        """
        tp = np.transpose(sample)
        self.x = tp[0]
        self.y = tp[1]
        self.z = tp[2]
        self.kappa = kappa
        self.sample = sample


class Coord3D:
    def __init__(self, x, y, z):
        """
        Store 3D coordinates.
        """
        self.x = x
        self.y = y
        self.z = z


