# November 2023
# Simone Mencarelli
# This file contains a class to calculate the array pattern of a conformal array, considering the
# rotation of the individual elements assuming elemental broadside being the normal to the conformal surface.

# %% imports
import numpy as np
import scipy as sp
from radartools.farField import UniformAperture
from numba import jit, prange, float64, types


# %% functions (for Jit optimization)
@jit(types.Tuple((float64[:, :], float64[:, :]))(float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :]),
     nopython=True, parallel=True)
def far_field_global_to_local_spherical_coordinates(theta_global, phi_global,
                                                    el_x, el_y, el_z):
    """
    Transform the global spherical coordinates to the local spherical coordinates
    :param theta_global: global theta coordinate SHALL be a 1d array [m_points] [rad]
    :param phi_global: global phi coordinate SHALL be a 1d array [m_points] [rad]
    :param el_x: local x versors SHALL be a 2d array [3, n_points]
    :param el_y: local y versors SHALL be a 2d array [3, n_points]
    :param el_z: local z versors SHALL be a 2d array [3, n_points]
    :return: m x n theta and phi coordinates in the local systems
    """
    # 1. transform the global spherical coordinates to the global cartesian coordinates
    r_global = np.ones_like(theta_global)
    x_global = r_global * np.sin(theta_global) * np.cos(phi_global)
    y_global = r_global * np.sin(theta_global) * np.sin(phi_global)
    z_global = r_global * np.cos(theta_global)
    # 2. initialize the local spherical coordinates
    theta_local = np.zeros((theta_global.size, el_x.shape[1]))
    phi_local = np.zeros_like(theta_local)
    # 3. for every point compute the local spherical coordinates
    for j in prange(el_x.shape[1]):
        # 3.1. create the rotation matrix from the global coordinate system to the local one for the current element
        R = np.empty((3, 3))
        R[:, 0] = el_x[:, j]
        R[:, 1] = el_y[:, j]
        R[:, 2] = el_z[:, j]
        # for every far field plane wave direction
        for i in prange(theta_global.size):
            # 3.2. transform the global cartesian coordinates to the local cartesian coordinates
            x_local, y_local, z_local = R @ np.array([x_global[i], y_global[i], z_global[i]])
            # 3.3. transform the local cartesian coordinates to the local spherical coordinates
            r = np.sqrt(x_local ** 2 + y_local ** 2 + z_local ** 2)
            theta_local[i, j] = np.where(r != 0, np.arccos(z_local / r), 0)
            phi_local[i, j] = np.where(r != 0, np.arctan2(y_local, x_local), 0)
    # 4. return the local spherical coordinates
    return theta_local, phi_local


# %% classes
class ConformalArray:
    def __init__(self, element_antenna,
                 points_x, points_y, points_z,
                 norm_x, norm_y, norm_z,
                 tan_x, tan_y, tan_z,
                 excitations, frequency,
                 c=299792458):
        """
        Create an Array object, defined by the sources locations and the complex excitations
        :param element_antenna: Aperture object
        :param points_x: x coordinate of points 1-d real array [m]
        :param points_y: y coordinate of points 1-d real array [m]
        :param points_z: z coordinate of points 1-d real array [m]
        :param norm_x: x coordinate of elements normal vector 1-d real array [m]
        :param norm_y: y coordinate of elements normal vector 1-d real array [m]
        :param norm_z: z coordinate of elements normal vector 1-d real array [m]
        :param tan_x: x coordinate of elements tangential vector 1-d real array [m] set as the x-axis of the individual radiating element
        :param tan_y: y coordinate of elements tangential vector 1-d real array [m] set as the x-axis of the individual radiating element
        :param tan_z: z coordinate of elements tangential vector 1-d real array [m] set as the x-axis of the individual radiating element
        :param excitations: complex excitation for each point
        :param frequency: operation frequency [Hz]
        :param c: optional, default 299792458 m/s
        :return: nothing
        """
        # frequency value [Hz]
        self.f = frequency
        # sources location in 3-d cartesian [m,m,m]
        self.points = np.array([points_x.reshape(-1), points_y.reshape(-1), points_z.reshape(-1)])
        # sources broadside normal vector in 3-d cartesian [m,m,m]
        self.el_z = np.array([norm_x.reshape(-1), norm_y.reshape(-1), norm_z.reshape(-1)])
        self.el_z = self.el_z / np.linalg.norm(self.el_z, axis=0)
        # sources tangential vector in 3-d cartesian [m,m,m]
        self.el_x = np.array([tan_x.reshape(-1), tan_y.reshape(-1), tan_z.reshape(-1)])
        self.el_x = self.el_x / np.linalg.norm(self.el_x, axis=0)
        # wavenumber absolute value
        self.k = 2 * np.pi * self.f / c
        # excitation factors
        self.excitations = excitations
        # array of element antennas
        self.element_antenna = element_antenna
        # compute and save the second tangential vector from tan and norm
        el_y = np.cross(self.el_z, self.el_x, axis=0)
        self.el_y = el_y / np.linalg.norm(el_y, axis=0)

    def far_field(self):
        pass


# %%
# main function
if __name__ == '__main__':
    # test the init function of the class
    # Reference uniform linear array
    lam = 3e8 / 10e9
    # element spacing
    dx = lam / 2
    # number of elements
    N = 13
    # x coordinates of the elements
    x = np.arange(N) * dx
    # y coordinates of the elements
    y = np.zeros_like(x)
    # element defined as a uniform aperture object
    radiator = UniformAperture(dx, dx, 10e9)
    # normal vectors of the elements
    norm_x = np.zeros_like(x)
    norm_y = np.zeros_like(x)
    norm_z = np.ones_like(x)
    # tangential vectors of the elements (x)
    tan_x = np.ones_like(x)
    tan_y = np.zeros_like(x)
    tan_z = np.zeros_like(x)
    # uniform excitation for the array
    exc = np.ones_like(x)
    # create array object
    uniform_array = ConformalArray(radiator, x, y, y, norm_x, norm_y, norm_z, tan_x, tan_y, tan_z, exc, 10e9)
    print("pippo")

    # %% test the far_field_global_to_local_spherical_coordinates function
    # define the angles for the far field, discretized in sin(theta) and phi
    theta = np.linspace(-np.pi / 2, np.pi / 2, 3)
    phi = np.ones_like(theta) * 0
    l_t, l_p = far_field_global_to_local_spherical_coordinates(theta, phi, uniform_array.el_x, uniform_array.el_y,
                                                               uniform_array.el_z)
    # works well for the uniform linear array case
    print("pippo")
