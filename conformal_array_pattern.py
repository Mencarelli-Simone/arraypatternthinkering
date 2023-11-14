# November 2023
# Simone Mencarelli
# This file contains a class to calculate the array pattern of a conformal array, considering the
# rotation of the individual elements assuming elemental broadside being the normal to the conformal surface.

# %% imports
import numpy as np
import scipy as sp
from radartools.farField import UniformAperture
from numba import jit, prange, float64, types
from numpy import sin, cos, tan


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
        for i in prange(theta_global.size):  # could be done using just array multiplication
            #                                  appropriately (but it doesn't work with jit that way)
            # 3.2. transform the global cartesian coordinates to the local cartesian coordinates
            x_local, y_local, z_local = R @ np.array([x_global[i], y_global[i], z_global[i]])
            # 3.3. transform the local cartesian coordinates to the local spherical coordinates
            r = np.sqrt(x_local ** 2 + y_local ** 2 + z_local ** 2)
            theta_local[i, j] = np.where(r != 0, np.arccos(z_local / r), 0)
            phi_local[i, j] = np.where(r != 0, np.arctan2(y_local, x_local), 0)
    # 4. return the local spherical coordinates
    return theta_local, phi_local


def far_field_local_to_global_spherical_coordinates(theta_local, phi_local, el_x, el_y, el_z, e_theta_local,
                                                    e_phi_local):
    """
    transforms the radiated Electric field in the local theta and phi coordinates to the global theta and phi
    coordinates for array pattern integration
    :param theta_local: theta coordinates in the local system SHALL be a 2d array [m_points, n_points] [rad]
    :param phi_local: phi coordinates in the local system SHALL be a 2d array [m_points, n_points] [rad]
    :param el_x: local x versors SHALL be a 2d array [3, n_points]
    :param el_y: local y versors SHALL be a 2d array [3, n_points]
    :param el_z: local z versors SHALL be a 2d array [3, n_points]
    :param e_theta_local: local theta component of the electric field SHALL be a 2d array [m_points, n_points]
    :param e_phi_local: local phi component of the electric field SHALL be a 2d array [m_points, n_points]
    :return:
    """
    # 0. Unravel the arrays for calculation
    original_shape = e_theta_local.shape
    e_t = e_theta_local.reshape(-1)
    e_p = e_phi_local.reshape(-1)
    theta_local = theta_local.reshape(-1)
    phi_local = phi_local.reshape(-1)

    # 1. transform the spherical local coordinates to the cartesian local coordinates
    e_x_local = cos(theta_local) * cos(phi_local) * e_t - sin(phi_local) * e_p
    e_y_local = sin(phi_local) * cos(theta_local) * e_t + cos(phi_local) * e_p
    e_z_local = -sin(theta_local) * e_t

    # 2. initialize the global cartesian coordinates
    e_x_global = np.zeros_like(e_x_local)
    e_y_global = np.zeros_like(e_y_local)
    e_z_global = np.zeros_like(e_z_local)

    # 3. reshape the local cartesian coordinates to the original format
    e_x_local = e_x_local.reshape(original_shape)
    e_y_local = e_y_local.reshape(original_shape)
    e_z_local = e_z_local.reshape(original_shape)

    # 3. for every radiator in the array compute the global cartesian coordinates transformation matrix
    for j in prange(el_x.shape[1]):
        # 3.1. create the rotation matrix from the local coordinate system to the global one for the current element
        R = np.empty((3, 3))
        R[:, 0] = el_x[:, j]
        R[:, 1] = el_y[:, j]
        R[:, 2] = el_z[:, j]
        R = R.T
        # 3.2. for every radiation angle compute the global cartesian coordinates
        for i in prange(original_shape[0]):
            # 3.3. transform the local cartesian coordinates to the global cartesian coordinates
            e_x_global[i, j], e_y_global[i, j], e_z_global[i, j] = R @ np.array(
                [e_x_local[i, j], e_y_local[i, j], e_z_local[i, j]])

# todo complete the function

    pass


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
        el_y = np.cross(self.el_z, self.el_x, axis=0)  # right-handed all right
        self.el_y = el_y / np.linalg.norm(el_y, axis=0)

    def far_field(self, theta, phi):
        """
        Returns the electric field radiation in terms of E theta and E phi at the specified theta and phi coordinates
        :param theta: theta coordinates meshgrid or 1-d vector [rad]
        :param phi: phi coordinates meshgrid or 1-d vector [rad]
        :return: E_theta, E_phi, same size as theta or phi [V/m]
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
        k *= self.k  # scaling
        # 1. transform the global spherical coordinates to the local spherical coordinates
        theta_local, phi_local = far_field_global_to_local_spherical_coordinates(theta, phi,
                                                                                 self.el_x, self.el_y, self.el_z)
        # 2. compute the far field of the individual elements
        E_theta, E_phi = self.element_antenna.mesh_E_field(theta_local, phi_local)
        # 3. Array factor transfer function matrix
        H = np.exp(1j * k.T @ self.points)
        pass

    def near_field(self, theta, phi):
        pass  # don't need it for the moment, might be useful for feed arrays in the future


# %%
# main function for some testing. not exhaustive, mainly for debugging
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
    theta = np.linspace(-np.pi / 2, np.pi / 2, 5)
    phi = np.ones_like(theta) * 0
    l_t, l_p = far_field_global_to_local_spherical_coordinates(theta, phi, uniform_array.el_x, uniform_array.el_y,
                                                               uniform_array.el_z)
    # works well for the uniform linear array case
    print("pippo")
