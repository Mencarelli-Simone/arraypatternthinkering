# November 2023
# Simone Mencarelli
# This file contains a class to calculate the array pattern of a conformal array, considering the
# rotation of the individual elements assuming elemental broadside being the normal to the conformal surface.
import matplotlib.pyplot as plt
# %% imports
import numpy as np
import scipy as sp
from radartools.farField import UniformAperture
from numba import jit, prange, float64, types, complex128
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
        R = R.T
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


@jit(types.Tuple((complex128[:, :], complex128[:, :], complex128[:, :]))
         (float64[:], float64[:],
          float64[:, :], float64[:, :],
          float64[:, :], float64[:, :], float64[:, :],
          complex128[:, :], complex128[:, :]), nopython=True, parallel=True)
def far_field_local_to_global_spherical_coordinates(theta_global, phi_global,
                                                    theta_local, phi_local,
                                                    el_x, el_y, el_z,
                                                    e_theta_local, e_phi_local):
    """
    transforms the radiated Electric field in the local theta and phi coordinates to the global theta and phi
    coordinates for array pattern integration
    :param theta_global: global theta coordinate radiation direction SHALL be a 1d array [m_points] [rad]
    :param phi_global: global phi coordinate radiation direction SHALL be a 1d array [m_points] [rad]
    :param theta_local: theta coordinates in the local system SHALL be a 2d array [m_points, n_points] [rad]
    :param phi_local: phi coordinates in the local system SHALL be a 2d array [m_points, n_points] [rad]
    :param el_x: local coordinate x versors SHALL be a 2d array [3, n_points]
    :param el_y: local coordinate y versors SHALL be a 2d array [3, n_points]
    :param el_z: local z coordinate versors SHALL be a 2d array [3, n_points]
    :param e_theta_local: local theta component of the electric field SHALL be a 2d array [m_points, n_points]
    :param e_phi_local: local phi component of the electric field SHALL be a 2d array [m_points, n_points]
    :return: e_r_global, e_theta_global, e_phi_global, same size as theta_global or phi_global [V/m]
    """
    # 0. Unravel the arrays for calculation
    original_shape = e_theta_local.shape
    e_t = np.ascontiguousarray(e_theta_local).reshape(-1)
    e_p = np.ascontiguousarray(e_phi_local).reshape(-1)
    theta_local = np.ascontiguousarray(theta_local).reshape(-1)
    phi_local = np.ascontiguousarray(phi_local).reshape(-1)
    phi_local = np.where(theta_local < 0, (phi_local + np.pi) % (2 * np.pi), phi_local) # doesn't change anything
    theta_local = np.abs(theta_local)

    # 1. transform the spherical local coordinates to the cartesian local coordinates vector field
    e_x_local = cos(theta_local) * cos(phi_local) * e_t - sin(phi_local) * e_p
    e_y_local = sin(phi_local) * cos(theta_local) * e_t + cos(phi_local) * e_p
    e_z_local = -sin(theta_local) * e_t

    # 2. reshape the local cartesian coordinates to the original format
    e_x_local = e_x_local.reshape(original_shape)
    e_y_local = e_y_local.reshape(original_shape)
    e_z_local = e_z_local.reshape(original_shape)

    # 2.1 initialize the global spherical coordinate output electric field
    e_r_global = np.zeros_like(e_x_local)  # it should be zero at the end, computing it just to see if its working
    e_theta_global = np.zeros_like(e_x_local)
    e_phi_global = np.zeros_like(e_x_local)

    # 3. for every radiator in the array compute the global cartesian coordinates transformation matrix
    for j in prange(el_x.shape[1]):  # n points (number of array cells)
        # 3.1. create the rotation matrix from the local coordinate system to the global one for the current element
        R = np.empty((3, 3), dtype=complex128)
        R[:, 0] = el_x[:, j]
        R[:, 1] = el_y[:, j]
        R[:, 2] = el_z[:, j]

        # 4. for every radiation angle:
        for i in prange(original_shape[0]):  # m points (number of radiation angles)
            # 4.1. transform the e field in local cartesian coordinates to the global cartesian coordinates e field
            E_cart = R @ np.array(
                [e_x_local[i, j], e_y_local[i, j], e_z_local[i, j]])  # is this right?
            # 4.2. transform the global cartesian coordinates to the global spherical coordinates electric field
            # spherical coordinates transformation matrix for current radiation angle
            S = np.array([[cos(phi_global[i]) * sin(theta_global[i]), sin(phi_global[i]) * sin(theta_global[i]),
                           cos(theta_global[i])],
                          [cos(phi_global[i]) * cos(theta_global[i]), sin(phi_global[i]) * cos(theta_global[i]),
                           -sin(theta_global[i])],
                          [-sin(phi_global[i]), cos(phi_global[i]), 0]], dtype=complex128)
            [e_r_global[i, j], e_theta_global[i, j], e_phi_global[i, j]] = S @ E_cart

    return e_r_global, e_theta_global, e_phi_global


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
        # # renormalise theta and phi
        phi = np.where(theta < 0, (phi - np.pi) % (np.pi * 2), phi)
        theta = np.abs(theta)
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
        E_theta, E_phi = self.element_antenna.mesh_E_field_theor(theta_local, phi_local)
        # 2.1 transform the far field to the global spherical coordinates
        E_theta, E_phi, E_r = far_field_local_to_global_spherical_coordinates(theta, phi, theta_local, phi_local,
                                                                              self.el_x, self.el_y, self.el_z,
                                                                              E_theta, E_phi)
        # 3. Array factor transfer function matrix
        H = np.exp(1j * k.T @ self.points)
        # 3.1 create the array factor matrix for the theta and phi components with amplitude
        H_theta = H * E_theta
        H_phi = H * E_phi
        # 3.2 save the array factor matrix for the theta and phi components
        self.H_theta = H_theta
        self.H_phi = H_phi
        # 4. compute the total electric field in the far field
        E_t = H_theta @ self.excitations
        E_p = H_phi @ self.excitations
        # 5 return the electric field in the far field
        return E_t.reshape(original_shape), E_p.reshape(original_shape)

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
    N = 7
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
    theta = np.linspace(-np.pi / 2, np.pi / 2, 21)
    phi = np.ones_like(theta) * 0
    phi = np.where(theta < 0, (phi - np.pi) % (2 * np.pi), phi)
    theta = np.abs(theta)
    l_t, l_p = far_field_global_to_local_spherical_coordinates(theta, phi, uniform_array.el_x, uniform_array.el_y,
                                                               uniform_array.el_z)
    # works well for the uniform linear array case
    print("pippo")

    # %% test the far_field_local_to_global_spherical_coordinates function
    # create a uniform aperture object to test the far field
    element_aperture = UniformAperture(dx, dx, 10e9)
    # evaluate the far field of the uniform aperture at the local theta and phi coordinates
    e_t, e_p = element_aperture.mesh_E_field(l_t, l_p)
    # transform the far field from local to global spherical coordinates
    el_r, el_t, el_p = far_field_local_to_global_spherical_coordinates(theta, phi, l_t, l_p,
                                                                       uniform_array.el_x, uniform_array.el_y,
                                                                       uniform_array.el_z, e_t, e_p)
    # DEBUG
    print('pippo')
    # the resulting el_t and el_p should be the same of e_t and e_p, just repeated for every element
    # e_r should be 0 or near 0

    # %% test the element field function
    # define the angles for the far field, discretized in sin(theta) and phi
    theta = np.linspace(-np.pi / 2, np.pi / 2, 21)
    phi = np.ones_like(theta) * 0
    phi = phi.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    # evaluate the far field of the uniform aperture at the local theta and phi coordinates
    e_t, e_p = element_aperture.mesh_E_field(theta, phi)
    radiatedPower = element_aperture.get_radiated_power()
    # plot the gain
    g = np.array(
        2 * np.pi * (np.abs(e_t) ** 2 + np.abs(e_p) ** 2) / (120 * np.pi * radiatedPower),
        dtype=float)
    fig, ax = plt.subplots(1)
    ax.plot(theta, g)
    plt.show()
    # plot e_theta
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_t) * np.sqrt(radiatedPower * element_aperture.eta))
    ax.plot(theta, np.angle(e_t))
    plt.show()
    # plot e_phi
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_p) * np.sqrt(radiatedPower * element_aperture.eta))
    ax.plot(theta, np.angle(e_p))
    plt.show()
    print("pippo")

    # %% test the theoretical element field function

    # define the angles for the far field, discretized in sin(theta) and phi
    theta = np.linspace(-np.pi / 2, np.pi / 2, 21)
    phi = np.ones_like(theta) * 0
    phi = np.where(theta < 0, (phi - np.pi) % (2 * np.pi), phi)
    theta = np.abs(theta)
    # evaluate the far field of the uniform aperture at the local theta and phi coordinates
    e_t, e_p = element_aperture.mesh_E_field_theor(theta, phi)
    radiatedPower = element_aperture.get_radiated_power()
    # plot the gain
    theta = np.linspace(-np.pi / 2, np.pi / 2, 21)
    g = np.array(
        2 * np.pi * (np.abs(e_t) ** 2 + np.abs(e_p) ** 2) / (120 * np.pi * radiatedPower),
        dtype=float)
    fig, ax = plt.subplots(1)
    ax.plot(theta, g)
    plt.show()
    # plot e_theta
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_t))
    ax.plot(theta, np.angle(e_t))
    plt.show()
    # plot e_phi
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_p))
    ax.plot(theta, np.angle(e_p))
    plt.show()
    print("pippo")

    # %% test the array far_field function
    # define the angles for the far field, discretized in sin(theta) and phi
    theta = np.linspace(-np.pi / 10, np.pi / 10, 11)
    phi = np.ones_like(theta) * 0
    phi = np.where(theta < 0, (phi + np.pi) % (2 * np.pi), phi)
    theta1 = np.abs(theta)
    # evaluate the far field of the array aperture at the local theta and phi coordinates
    e_t, e_p = uniform_array.far_field(theta1, phi)
    radiatedPower = np.sum(np.abs(exc) ** 2 * uniform_array.element_antenna.get_radiated_power() ** 2)
    # plot the gain
    g = np.array(
        2 * np.pi * (np.abs(e_t) ** 2 + np.abs(e_p) ** 2) / (120 * np.pi * radiatedPower),
        dtype=float)
    fig, ax = plt.subplots(1)
    ax.plot(theta, g)
    plt.show()
    print("pippo")  # todo debug the theoretial efield function. is clearly not working as it should.
