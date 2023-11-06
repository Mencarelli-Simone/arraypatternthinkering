# Simone Mencarelli, November 2023
# this file contains the array factor functions for a generalized scatterers cloud
# Note, for conformal array,this model is not valid as rotation of the individual element patterns is also
# required.

# %% imports
import numpy as np
import scipy as sp
from radartools.utils import *


# %% functions (for Jit optimization)


# %% classes

class Array:
    def __int__(self, points_x, points_y, points_z, excitations, frequency, c=299792458):
        """
        Create an Array object, defined by the sources locations and the complex excitations
        :param points_x: x coordinate of points 1-d real array [m]
        :param points_y: y coordinate of points 1-d real array [m]
        :param points_z: z coordinate of points 1-d real array [m]
        :param excitations: complex excitation for each point
        :param frequency: operation frequency [Hz]
        :param c: optional, default 299 792 458 m/s
        :return: nothing
        """
        # frequency value [Hz]
        self.f = frequency
        # sources location in 3-d cartesian [m,m,m]
        self.points = np.array([[points_x.reshape(-1)], [points_y.reshape(-1)], [points_z.reshape(-1)]])
        # wavenumber absolute value
        self.k = 2 * np.pi * self.f / c
        # excitation factors
        self.excitations = excitations

    def factor(self, theta, phi):
        """
        Returns the array factor at the specified theta and phi coordinates
        :param theta: theta coordinates meshgrid or 1-d vector [rad]
        :param phi: phi coordinates meshgrid or 1-d vector [rad]
        :return: complex array factor, same size as theta or phi [neper]
        """
        original_shape = theta.shape
        # unravel the arrays for calculation
        theta = theta.reshape(-1)
        phi = phi.reshape(-1)
        # range coordinate
        r = np.ones_like(theta)
        # polar to cartesian coordinates conversion, vector plane wave (spatial frequency)
        k = sph2cart(np.array([r, theta, phi])).T # column
        k *= self.k
        # copy the k column as many times as self.points is long

        scalar = np.dot(k, self)