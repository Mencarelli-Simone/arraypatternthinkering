# Simone Mencarelli, November 2023
# this file contains the array factor functions for a generalized scatterers cloud
# Note, for conformal array,this model is not valid as rotation of the individual element patterns is also
# required.

# %% imports
import numpy as np
import scipy as sp
from radartools.utils import *
import matplotlib.pyplot as plt


# %% functions (for Jit optimization)


# %% classes

class AntennaArray:
    def __init__(self, points_x, points_y, points_z, excitations, frequency, c=299792458):
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
        self.points = np.array([points_x.reshape(-1), points_y.reshape(-1), points_z.reshape(-1)])
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
        kx = r * np.sin(theta) * np.cos(phi)
        ky = r * np.sin(theta) * np.sin(phi)
        kz = r * np.cos(theta)
        k = np.array([kx, ky, kz])
        k *= self.k
        # phase of each source, calculated as the dot product of the spatial frequency k
        # with the array points
        phase = k.T @ self.points
        # array factor
        af = np.sum(self.excitations * np.exp(1j * phase), axis=1)
        # return to original shape
        return af.reshape(original_shape)

    def near_field_projection(self, points_x, points_y, points_z):
        """
        Returns the near field projection of the array at the specified points of a second
        virtual array as complex excitation values for the virtual array geometry passed as input
        :param points_x: x coordinates meshgrid or 1-d vector [m]
        :param points_y: y coordinates meshgrid or 1-d vector [m]
        :param points_z: z coordinates meshgrid or 1-d vector [m]
        :return: complex array excitations for the virtual array, same size as points_x, points_y, points_z [neper]
        """
        # near field point locations in 3-d cartesian [m,m,m]
        # relative range vector matrix generation:
        # 1. repeat the points array as many times as the number of sources
        Px = np.repeat(points_x.reshape(-1, 1), self.points.shape[1], axis=1)
        Py = np.repeat(points_y.reshape(-1, 1), self.points.shape[1], axis=1)
        Pz = np.repeat(points_z.reshape(-1, 1), self.points.shape[1], axis=1)
        # 2. repeat the self.points array as many times as the number of points
        Sx = np.repeat(self.points[0, :].reshape(1, -1), points_x.shape[0], axis=0)
        Sy = np.repeat(self.points[1, :].reshape(1, -1), points_y.shape[0], axis=0)
        Sz = np.repeat(self.points[2, :].reshape(1, -1), points_z.shape[0], axis=0)
        # 3. calculate the relative range vector matrix
        R = np.array([Px - Sx, Py - Sy, Pz - Sz])
        # 4. calculate the relative range vector matrix norm
        R_norm = np.linalg.norm(R, axis=0)
        # find the transfer function matrix
        H = np.exp(-1j * self.k * R_norm) / R_norm * 1 / (4 * np.pi)
        # calculate the excitations
        excitations = H @ self.excitations
        # return the excitations
        return excitations


if __name__ == '__main__':
    # %% Test 1: 1-d array, uniform excitation
    # wavelength
    lam = 3e8 / 10e9
    # spacing
    dx = lam / 2
    # 10 lambda array
    x = np.arange(-5 * lam, 5 * lam, dx)
    # excitation
    ex = np.ones_like(x)
    # array object
    array = AntennaArray(x, np.zeros_like(x), np.zeros_like(x), ex, 10e9)
    # theta and phi coordinates
    theta = np.linspace(-np.pi / 2, np.pi / 2, 360)
    phi = np.ones_like(theta) * 0
    # array factor
    af = array.factor(theta, phi)
    # plot
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / np.pi, np.abs(af))
    plt.show()

    # Test 2: 1-d array, uniform excitation but in the y-z plane
    # wavelength
    lam = 3e8 / 10e9
    # spacing
    dy = lam / 2
    # 10 lambda array
    y = np.arange(-5 * lam, 5 * lam, dy)
    # excitation
    ey = np.ones_like(y)
    # array object
    array = AntennaArray(np.zeros_like(y), y, np.zeros_like(y), ey, 10e9)
    # theta and phi coordinates
    theta = np.linspace(-np.pi / 2, np.pi / 2, 360)
    phi = np.ones_like(theta) * np.pi / 2
    # array factor
    af = array.factor(theta, phi)
    # plot
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / np.pi, np.abs(af))
    plt.show()

    # %% Test 3: Near field projection
    # wavelength
    lam = 3e8 / 10e9
    # spacing
    dx = lam / 2
    # 10 lambda array
    x = np.arange(-5 * lam, 5 * lam, dx)
    # excitation
    ex = np.ones_like(x)
    # array object
    array = AntennaArray(x, np.zeros_like(x), np.zeros_like(x), ex, 10e9)
    # near field points
    nx = np.arange(-5 * lam, 5 * lam, dx)
    ny = np.ones_like(nx)
    nz = np.zeros_like(nx)
    # near field projection
    exc = array.near_field_projection(nx, ny, nz)
    # plot
    fig, ax = plt.subplots(1)
    ax.plot(nx, np.abs(exc))
    ax.plot(nx, np.angle(exc))
    plt.show()

    # %% New array with the near field projection excitations
    # conservation of power
    exc *= np.sqrt(np.sum(np.abs(ex) ** 2) / np.sum(np.abs(exc) ** 2))
    # array object
    array2 = AntennaArray(nx, ny, nz, np.abs(exc), 10e9)
    # theta and phi coordinates
    theta = np.linspace(-np.pi / 2, np.pi / 2, 360)
    phi = np.ones_like(theta) * 0\
    # array factor\
    af1 = array2.factor(theta, phi)
    # plot
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / np.pi, np.abs(af1))
    ax.plot(theta * 180 / np.pi, np.abs(af))
    plt.show()
