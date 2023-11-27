# Simone Mencarelli
# November 2023
# This file contains a class to describe the interface for a feed antenna to be used in a reflectarray
# the class might be extended to include specific feeds or file-defined feed radiation patterns.
# %% includes
import numpy as np
from numpy import pi, sin, cos, tan, arcsin, exp


# %% Functions

# %% Feed antenna class
class FeedAntenna():
    def __init__(self, x, y, z, x_norm, y_norm, z_norm, x_tan, y_tan, z_tan, frequency, c=299792458):
        """
        Initialize the feed antenna

        :param x: x coordinate of the feed antenna
        :param y: y coordinate of the feed antenna
        :param z: z coordinate of the feed antenna
        :param x_norm: x component of the feed antenna radiation normal, local z-axis
        :param y_norm: y component of the feed antenna radiation normal, local z-axis
        :param z_norm: z component of the feed antenna radiation normal, local z-axis
        :param x_tan: x component of the feed antenna tangent local x-axis
        :param y_tan: y component of the feed antenna tangent local x-axis
        :param z_tan: z component of the feed antenna tangent local x-axis
        :param frequency: frequency of operation
        :param c: speed of light, optional, default 299792458 m/s
        :return: None
        """
        # frequency of operation
        self.freq = frequency
        # free space impedance
        self.eta = 376.7303136686
        # compute the wavelength
        self.wavelength = c / frequency
        # position of the feed antenna
        self.pos = np.array([x, y, z])
        # radiation normal of the feed antenna
        self.z = np.array([x_norm, y_norm, z_norm])
        # tangent of the feed antenna
        self.x = np.array([x_tan, y_tan, z_tan])
        # y-axis of the feed antenna
        self.y = np.cross(self.z_f, self.x_f)

    def e_field(self, r, theta, phi):
        """
        Compute the electric field radiated by the feed antenna at the specified point
        :param r: feed local radius
        :param theta: feed local theta
        :param phi: feed local phi
        :return: e theta e phi
        """
        # dummy unitary amplitude, phase dependent on r
        amplitude = 1
        phase = exp(-1j * 2 * pi * r / self.wavelength)
        # x polarization, no dependence on r
        e_theta = amplitude * phase * cos(phi)
        e_phi = amplitude * phase * sin(phi)
        return np.zeros_like(e_theta), e_theta, e_phi

    def e_field_cartesian(self, x, y, z):
        """
        Compute the electric field radiated by the feed antenna at the specified point in the local cartesian system
        :param x: lcs x coordinate vector
        :param y: lcs y coordinate vector
        :param z: lcs z coordinate vector
        :return:
        """
        # save the shape of the input vectors
        shape = x.shape
        # reshape the input vectors
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        # from the local cartesian to the local spherical
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.where(r != 0, np.arccos(z / r), 0)
        phi = np.where(r != 0, np.arctan2(y, x), 0)
        # compute the electric field
        e_r, e_theta, e_phi = self.e_field(r, theta, phi)
        # from the local spherical to the local cartesian
        e_x = e_r * sin(theta) * cos(phi) + e_theta * cos(theta) * cos(phi) - e_phi * sin(phi)
        e_y = e_r * sin(theta) * sin(phi) + e_theta * cos(theta) * sin(phi) + e_phi * cos(phi)
        e_z = e_r * cos(theta) - e_theta * sin(theta)
        # reshape the output vectors
        e_x = e_x.reshape(shape)
        e_y = e_y.reshape(shape)
        e_z = e_z.reshape(shape)
        return e_x, e_y, e_z

    def e_field_gcs(self, x, y, z):
        """
        Compute the electric field radiated by the feed antenna at the specified point in the global cartesian system
        :param x: gcs x coordinate vector
        :param y: gcs y coordinate vector
        :param z: gcs z coordinate vector
        :return: e_x, e_y, e_z
        """
        # save the shape of the input vectors
        shape = x.shape
        # reshape the input vectors
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        # from the global cartesian to the local cartesian
        # rotation matrix from the global to the local
        R = np.array([self.x, self.y, self.z]).T
        # local cartesian coordinates
        x_lcs, y_lcs, z_lcs = R @ (np.array([x, y, z]) - self.pos)  # todo debug
        # compute the electric field in the local cartesian system
        e_x, e_y, e_z = self.e_field_cartesian(x_lcs, y_lcs, z_lcs)
        # reshape the output vectors
        e_x = e_x.reshape(shape)
        e_y = e_y.reshape(shape)
        e_z = e_z.reshape(shape)
        return e_x, e_y, e_z
