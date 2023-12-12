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
from numpy import sin, cos, tan, arcsin, pi
import mayavi.mlab as ml


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
    phi_local = np.where(theta_local < 0, (phi_local + np.pi) % (2 * np.pi), phi_local)  # doesn't change anything
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
        # from LCS to GCS no need to transpose. this is verified by the wireframe plot for the elements (uses same R)
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


# %% Functions (general utilities)
def create_cylindrical_array(length, width, angular_section, dx, freq):
    """
    Creates a cylindrical array of uniformly spaced square elements, each one a UniformAperture object
    the length and width are rounded to the closest approximation, messages are printed to the console.
    :param length: aperture length [m]
    :param width: aperture width (circle section chord) [m]
    :param angular_section: angular section of the cylinder [rad]
    :param dx: element spacing [m]
    :param frequency: operation frequency [Hz]
    :return: ConformalArrayobject, radius of the cylinder
    """
    L = length
    W = width
    tc = angular_section
    dc = dx
    # radius
    r = W / (sin(tc / 2) * 2)
    # For the length is easy to define the number of elements
    # align aperture size to a multiple of dx dy
    L = np.ceil(L / dx) * dx
    # number of elements in length
    Nx = int(L / dx)
    # make it odd to have a central element
    if Nx % 2 == 0:
        Nx += 1
    # actual length
    L = Nx * dx
    # for the width we first find the closest approximation of the number of elements
    dt = 2 * arcsin(dc / (2 * r))
    # the number of radial segments is then (defect approximation)
    Nt = int(tc / dt)
    # make it odd to have a central element
    if Nt % 2 == 0:
        Nt += 1
    # actual circular section
    tc = Nt * dt
    # actual width
    W = 2 * r * sin(tc / 2)
    # print actual length and width limiting the number of digits to two after the point
    print(f'Aperture size: {L:.4f} x {W:.4f} m')
    # print actual number of elements
    print(f'Number of elements: {Nx} x {Nt}')
    # print actual subtended angle and radius
    print(f'Subtended angle: {tc * 180 / pi :.4f} deg')
    print(f'Radius: {r:.4f} m')

    # %% Array geometry
    # 1. location of elements in the array, separating the problem in length and section
    # length
    xc = np.arange(0, Nx) * dx - L / 2 + dx / 2
    # theta
    t = np.arange(0, Nt) * dt - tc / 2 + dt / 2
    # xc repeats along the theta axis
    xc_mesh, t_mesh = np.meshgrid(xc, t)
    # individual points y and z coordinates
    yc_mesh = r * sin(t_mesh)
    zc_mesh = r - r * cos(t_mesh)
    # 2. radiation normal of the elements in the circular section
    norm_x = np.zeros_like(t_mesh)
    # the norm lies in the y-z plane
    norm_y = -sin(t_mesh)
    norm_z = cos(t_mesh)
    # 3. x-axis tangent to the element, i.e. global x-axis
    tan_x = np.ones_like(t_mesh)
    tan_y = np.zeros_like(t_mesh)
    tan_z = np.zeros_like(t_mesh)

    # %% create the array
    # uniform excitation vector
    excitation = np.ones_like(t_mesh)
    # create half wavelength elemental aperture
    element = UniformAperture(dx, dx, freq)
    # create the array
    array = ConformalArray(element,
                           xc_mesh, yc_mesh, zc_mesh,
                           norm_x, norm_y, norm_z,
                           tan_x, tan_y, tan_z,
                           excitation, freq)
    # %% set the element wireframe
    # %% elemental wireframe
    wf_x = np.array([-dx / 2, dx / 2, dx / 2, -dx / 2, -dx / 2])
    wf_y = np.array([-dx / 2, -dx / 2, dx / 2, dx / 2, -dx / 2])
    wf_z = np.array([0, 0, 0, 0, 0])
    array.set_element_wireframe(wf_x, wf_y, wf_z)
    # %% element surface
    # create the surface
    return array, r

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
        :param points_x: x coordinate of points 1-d or 2-d real array [m]
        :param points_y: y coordinate of points 1-d or 2-d real array [m]
        :param points_z: z coordinate of points 1-d or 2-d real array [m]
        :param norm_x: x coordinate of elements normal vector 1-d or 2-d real array [m]
        :param norm_y: y coordinate of elements normal vector 1-d or 2-d real array [m]
        :param norm_z: z coordinate of elements normal vector 1-d or 2-d real array [m]
        :param tan_x: x coordinate of elements tangential vector 1-d or 2-d real array [m] set as the x-axis of the individual radiating element
        :param tan_y: y coordinate of elements tangential vector 1-d or 2-d real array [m] set as the x-axis of the individual radiating element
        :param tan_z: z coordinate of elements tangential vector 1-d or 2-d real array [m] set as the x-axis of the individual radiating element
        :param excitations: complex excitation for each point
        :param frequency: operation frequency [Hz]
        :param c: optional, default 299792458 m/s
        :return: nothing
        """
        # frequency value [Hz]

        self.f = frequency
        # meshgrid shape
        self.shape = points_x.shape
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
        self.excitations = excitations.reshape(-1)
        # array of element antennas
        self.element_antenna = element_antenna
        # compute and save the second tangential vector from tan and norm
        el_y = np.cross(self.el_z, self.el_x, axis=0)  # right-handed all right
        self.el_y = el_y / np.linalg.norm(el_y, axis=0)
        # graphic
        self.element_wireframe = None  # wireframe for a single element
        self.element_surface = None  # surface for a single element
        self.element_surfaces = None  # list of surface objects (mayavi objects)
        self.element_wires = None  # many lines single mayavi object for rotated elemental wireframes

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
        E_r, E_theta, E_phi = far_field_local_to_global_spherical_coordinates(theta, phi, theta_local, phi_local,
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

    def plot_points(self, ax, **args):
        """
        Plots the points of the array on the specified axis
        :param ax: axis object
        :param args: optional, plot arguments
        :return: nothing
        """
        if args is None:
            args = {}
        ax.scatter(self.points[0, :], self.points[1, :], self.points[2, :], **args)

    def plot_lcs(self, ax, length=0.01, **args):
        """
        Plots the local coordinate system of each element on the specified axis as quivers
        :param ax:
        :param length:
        :param args:
        :return:
        """
        if args is None:
            args = {}
        # plot the local coordinate system
        ax.quiver(self.points[0, :], self.points[1, :], self.points[2, :],
                  self.el_x[0, :], self.el_x[1, :], self.el_x[2, :], length=length, color='r', **args)
        ax.quiver(self.points[0, :], self.points[1, :], self.points[2, :],
                  self.el_y[0, :], self.el_y[1, :], self.el_y[2, :], length=length, color='g', **args)
        ax.quiver(self.points[0, :], self.points[1, :], self.points[2, :],
                  self.el_z[0, :], self.el_z[1, :], self.el_z[2, :], length=length, color='b', **args)

    # should add a MAYAVI compatible plotter equivalent
    def plot_points_mayavi(self, **args):
        """
        Plots the points of the array using MayaVi points3d function
        :param args:
        :return:
        """
        if args is None:
            args = {}
        ml.points3d(self.points[0, :], self.points[1, :], self.points[2, :], **args)

    def plot_lcs_mayavi(self, length=0.01, **args):
        """
        Plots the local coordinate system of each element on the specified axis as quivers
        :param length: versors length
        :param args:
        :return:
        """
        if args is None:
            args = {}
        # plot the local coordinate system
        a = ml.quiver3d(self.points[0, :], self.points[1, :], self.points[2, :],
                        self.el_x[0, :], self.el_x[1, :], self.el_x[2, :], scale_factor=length,
                        color=(1, 0, 0), **args)
        b = ml.quiver3d(self.points[0, :], self.points[1, :], self.points[2, :],
                        self.el_y[0, :], self.el_y[1, :], self.el_y[2, :], scale_factor=length,
                        color=(0, 1, 0), **args)
        c = ml.quiver3d(self.points[0, :], self.points[1, :], self.points[2, :],
                        self.el_z[0, :], self.el_z[1, :], self.el_z[2, :], scale_factor=length,
                        color=(0, 0, 1), **args)
        return a, b, c

    def set_element_wireframe(self, points_x, points_y, points_z):
        """
        Sets the element wireframe for the plot
        :param points_x: element wireframe edges in lcs x coordinates
        :param points_y: element wireframe edges in lcs y coordinates
        :param points_z: element wireframe edges in lcs z coordinates
        :return:
        """
        self.element_wireframe = np.array([points_x, points_y, points_z])

    def draw_elements_mayavi(self, **args):
        """
        Draws the elements of the array using MayaVi
        :param args: passed to ml.pipeline.surface
        :return:
        """
        # adapted from the example "plotting_many_lines.py" from the mayavi documentation
        # https://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
        # lists of lines
        points_x = list()
        points_y = list()
        points_z = list()
        connections = list()
        # number of points per line
        N = self.element_wireframe.shape[1]
        # The index of the current point in the total amount of points
        index = 0
        # for every element in the array, apply the transformation matrix to the element wireframe
        for i in range(self.points.shape[1]):
            # create the transformation matrix
            R = np.empty((3, 3))
            R[:, 0] = self.el_x[:, i]
            R[:, 1] = self.el_y[:, i]
            R[:, 2] = self.el_z[:, i]
            # this is a LCS to GCS transformation matrix
            # transform the element wireframe, repeat self.points[:, i] for every point in the wireframe
            x, y, z = R @ self.element_wireframe + np.repeat(self.points[:, i].reshape(-1, 1),
                                                             self.element_wireframe.shape[1],
                                                             axis=1)
            # append the points to the lists
            points_x.append(x.reshape(-1))
            points_y.append(y.reshape(-1))
            points_z.append(z.reshape(-1))
            # create the connections and append them to the list
            connections.append(np.vstack(
                [np.arange(index, index + N - 1.5),
                 np.arange(index + 1, index + N - .5)]
            ).T)
            index += N

        # Now collapse all positions, scalars and connections in big arrays
        x = np.hstack(points_x)
        y = np.hstack(points_y)
        z = np.hstack(points_z)
        s = np.ones_like(z)
        connections = np.vstack(connections)
        # Create the points
        src = ml.pipeline.scalar_scatter(x, y, z, s)
        # Connect them
        src.mlab_source.dataset.lines = connections
        src.update()
        # The stripper filter cleans up connected lines
        lines = ml.pipeline.stripper(src)  # this line breaks windows in debug mode
        # Finally, display the set of lines
        wire = ml.pipeline.surface(lines, **args)
        self.element_wires = wire
        return wire

    def set_element_surface(self, x, y, scalar):
        """
        Sets the element surface for the plot
        :param x: element surface x coordinates 1-d or 2-d real array [m]
        :param y: element surface y coordinates 1-d or 2-d real array [m]
        :param scalar: element surface scalar value 2-d array [m,m]
        :return:
        """
        if x.shape != scalar.shape:
            x, y = np.meshgrid(x.reshape(-1), y.reshape(-1))
        self.element_surface = np.array([x, y, scalar])  # this is going to be a 3-d array

    def draw_element_surfaces_mayavi(self, mplcmap='jet', parameter='angle', **args):
        """
        Draws the elements of the array using MayaVi filling the surface using a parametric colormap
        :param args:
        :return:
        """
        cmap = plt.get_cmap(mplcmap)
        # if parameter is an array
        if isinstance(parameter, np.ndarray):
            parameter = parameter.reshape(-1)
        else:
            if parameter == 'angle':
                parameter = (np.angle(self.excitations) % (2 * np.pi)) / (2 * np.pi)
            elif parameter == 'abs':
                parameter = np.abs(self.excitations) / np.max(np.abs(self.excitations))

        # for every element in the array, apply the transformation matrix to the element surface
        self.element_surfaces = []
        for i in range(self.points.shape[1]):
            # create the transformation matrix
            R = np.empty((3, 3))
            R[:, 0] = self.el_x[:, i]
            R[:, 1] = self.el_y[:, i]
            R[:, 2] = self.el_z[:, i]
            # this is a LCS to GCS transformation matrix
            # transform the element surface, repeat self.points[:, i] for every point in the surface
            sx = self.element_surface[0, :, :].reshape(-1)
            sy = self.element_surface[1, :, :].reshape(-1)
            sz = self.element_surface[2, :, :].reshape(-1)
            shape = self.element_surface[0, :, :].shape
            x, y, scalar = R @ np.array([sx, sy, sz]) + np.repeat(self.points[:, i].reshape(-1, 1),
                                                                  sx.shape[0],
                                                                  axis=1)
            # create the surface
            surf = ml.mesh(x.reshape(shape), y.reshape(shape), scalar.reshape(shape),
                           color=cmap(parameter[i])[0:3], **args)
            # src = ml.pipeline.array2d_source(x.reshape(shape), y.reshape(shape), scalar.reshape(shape))
            # surf = ml.pipeline.surface(src,
            #                     color=cmap(parameter[i])[0:3], **args)
            self.element_surfaces.append(surf)


    def radiated_power(self):
        """
        Returns the radiated power of the array
        :return: radiated_power
        """
        radiated_power = np.sum(np.abs(self.excitations) ** 2 * self.element_antenna.get_radiated_power())
        return radiated_power

    def near_field(self, theta, phi):
        pass  # don't need it for the moment, might be useful for feed arrays in the future


# %%
# main function for some testing. not exhaustive, mainly for debugging
if __name__ == '__main__':
    # test the init function of the class
    # Reference uniform linear array
    lam = 3e8 / 10e9
    # element spacing
    dx = lam / 8
    # number of elements
    N = 70
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
    theta = np.linspace(-np.pi / 2, np.pi / 2, 30601)
    phi = np.ones_like(theta) * 0
    phi = np.where(theta < 0, (phi + np.pi) % (2 * np.pi), phi)
    theta1 = np.abs(theta)
    # evaluate the far field of the array aperture at the local theta and phi coordinates
    e_t, e_p = uniform_array.far_field(theta1, phi)
    radiatedPower = uniform_array.radiated_power()
    # plot the gain
    g = np.array(
        2 * np.pi * (np.abs(e_t) ** 2 + np.abs(e_p) ** 2) / (uniform_array.element_antenna.eta * radiatedPower),
        dtype=float)
    fig, ax = plt.subplots(1)
    ax.plot(theta, g)
    plt.show()
    print("pippo")
