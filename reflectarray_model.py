# Simone Mencarelli
# December 2023
# This file contains two classes:  a model for the reflectarray cell
# (to relate angle of incidence to reflected element excitation electric field)
# and a model for the reflectarray (to combine feed, cell and conformal array models).
# This approach to the modeling of the reflectarray antenna, is equivalent to the "aperture" approach
# described in the book :
# [1]   Reflectarray Antennas: Theory, Designs, and Applications, First Edition.
#       Payam Nayeri, Fan Yang, and Atef Z. Elsherbeni.z

# %% includes
import numpy as np
from numpy import pi, sin, cos, tan, arcsin
import matplotlib.pyplot as plt
from conformal_array_pattern import ConformalArray
from feed_antenna_pattern import FeedAntenna
from conformal_array_pattern import create_cylindrical_array
import matplotlib
import mayavi
from mayavi import mlab as ml

matplotlib.use('Qt5Agg')


# %% Functions

# %% Classes
# reflectarray cell model
class RACell:
    # constructor
    def __init__(self):
        """
        Create an ideal reflectarray cell model.
        in future iterations this shall point to a database of cells
        and an interpolation mathod shall be provided
        """
        pass

    def scattering_matrix(self, theta_inc, phi_inc, phase_shift):
        """
        Compute the scattering matrix of the cell provided the
        incidence angle and the phase shift of the element (i.e. the element index)
        This method shall be the interface to the cell database for the RA object
        :param theta_inc: theta incidence angle LCS of the element, single value or [N]
        :param phi_inc: phi incidence angle LCS of the element, same length of theta_inc
        :param phase_shift: phase shift of the element  single value or [N]
        :return: scattering matrix [2,2,N]
        """
        # for the ideal element the scattering matrix is a diagonal matrix with the phase shift
        # on the diagonal corresponding to the input parameter
        # if phase shift is an index
        if type(phase_shift) is int:
            length = 1
        else:
            length = len(phase_shift)
        gamma = np.zeros([2, 2, length], dtype=complex)
        gamma[0, 0, :] = np.exp(1j * phase_shift)
        gamma[1, 1, :] = np.exp(1j * phase_shift)
        # the reflected field is to be calculated according to [1] eq. 4.12
        # | Erx |   | gamma_11 gamma_12 | | Eix |
        # |     | = |                   | |     |
        # | Ery |   | gamma_21 gamma_22 | | Eiy |
        return gamma.squeeze()


# reflectarray model
class ReflectArray:
    # constructor
    def __init__(self, cell: RACell, feed: FeedAntenna, array: ConformalArray):
        """
        Create a reflectarray antenna model.
        :param cell: reflectarray cell model
        :param feed: feed antenna model
        :param array: conformal array model
        """
        # store the models
        self.cell = cell
        self.feed = feed
        self.array = array
        # incident fields gcs
        self.Ex_i = None
        self.Ey_i = None
        self.Ez_i = None
        # incident fields element by element lcs
        self.Ex_l = None
        self.Ey_l = None
        self.Ez_l = None
        # incidence angle for each element
        self.theta_inc = None
        self.phi_inc = None

    def compute_incident_tangential_field(self):
        # This method shall be jitted to run in parallel
        # 1. get global x, y, z e field for each element in the array
        Ex, Ey, Ez = self.feed.e_field_gcs(self.array.points[0], self.array.points[1], self.array.points[2])
        # save the gcs incident field
        self.Ex_i = Ex
        self.Ey_i = Ey
        self.Ez_i = Ez
        # initialise the local e field
        Ex_l = np.zeros_like(Ex)
        Ey_l = np.zeros_like(Ey)
        Ez_l = np.zeros_like(Ez)
        # 2. compute the local x, y, z e field for each element in the array
        for i in range(self.array.points.shape[1]):
            # 2.1 rotation matrix
            # create the transformation matrix
            R = np.empty((3, 3))
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            R = R.T  # from gcs to lcs you need the transpose
            # 2.2 compute the local e field
            Ex_l[i], Ey_l[i], Ez_l[i] = R @ np.array([Ex[i], Ey[i], Ez[i]])
        self.Ex_l = Ex_l
        self.Ey_l = Ey_l
        self.Ez_l = Ez_l
        return Ex_l, Ey_l, Ez_l

    def compute_elements_angle_of_incidence(self):
        """
        compute the angle of incidence for each element in the array
        :return theta_inc, phi_inc: angle of incidence for each element in the array
        """
        # initialise the angles
        theta_inc = np.zeros(self.array.points.shape[1])
        phi_inc = np.zeros(self.array.points.shape[1])
        # compute the angles
        for i in range(self.array.points.shape[1]):
            # convert the feed position in the lcs of the element
            # create the transformation matrix
            R = np.empty((3, 3))
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            # compute the feed position in the lcs of the element
            feed_pos_lcs = R.T @ (np.array([self.feed.x, self.feed.y, self.feed.z]) - self.array.points[:, i])
            # convert to spherical coordinates
            theta_inc[i] = arcsin(feed_pos_lcs[2] / np.linalg.norm(feed_pos_lcs))
            phi_inc[i] = np.arctan2(feed_pos_lcs[1], feed_pos_lcs[0])
    def collimate_beam(self, theta_broadside, phi_broadside):
        # returns the required phase shifts to collimate the beam
        pass

    def compute_reflected_tangential_field(self):
        pass

    def excite_conformal_array(self):
        pass

    def co_pol(self, theta, phi):
        pass

    def cross_pol(self, theta, phi):
        pass

    # graphics
    def draw_reflectarray(self):
        # draw the array
        self.array.draw_elements_mayavi()
        # draw the feed
        self.feed.draw_feed(scale=1)

    def draw_tangential_e_field(self, **kwargs):
        """
        Draw the tangential electric field on the reflectarray surface for every element
        it is assumed the incident e field is already computed and stored in 1
        :param kwargs: mayavi quiver3d kwargs
        :return:
        """
        # draw the e field with mayavi for every point
        ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], np.abs(self.Ex_i),
                    np.abs(self.Ey_i), np.abs(self.Ez_i), **kwargs)


# %% test code
# main
if __name__ == "__main__":
    # RA cell test
    # unitary incidence angles
    theta_inc = pi / 2
    phi_inc = 0
    # phase shift
    phase_shift = 0
    # create the cell
    cell = RACell()
    # compute the scattering matrix
    gamma = cell.scattering_matrix(theta_inc, phi_inc, phase_shift)
    # print the results
    print('gamma = \n', gamma)
    # array of phase shifts
    phase_shift = np.linspace(0, 2 * pi, 5)
    # compute the scattering matrix
    gamma = cell.scattering_matrix(theta_inc, phi_inc, phase_shift)
    # print the results liminting the number of decimals
    print('gamma = \n', np.round(gamma, 3))

    # %% RA test incident tangential field computation and visualization
    # geometry of the test array: cylindirical reflectarray, feed placed at the focus of the cylinder at
    # half the length of the reflectarray.
    # antenna parameters
    freq = 10e9
    wavelength = 3e8 / freq
    # elements spacing (square elements)
    dx = wavelength / 2
    # aperture size
    L = 2
    W = 0.3
    # circular section angle
    tc = 90 * pi / 180
    ## create the array
    array, radius = create_cylindrical_array(L, W, tc, dx, freq)
    ## create the feed
    # position the feed at the focus of the cylinder on the z axis pointing down
    x = 0
    y = 0
    z = 1  # m
    feed = FeedAntenna(x, y, z, 0, 0, -1, 1, 0, 0, freq)
    ## create ra cell
    cell = RACell()
    # create the reflectarray
    reflectarray = ReflectArray(cell, feed, array)
    # display the RA in mayavi
    ml.figure(2, bgcolor=(0, 0, 0))
    ml.clf()
    # draw the array
    reflectarray.array.draw_elements_mayavi(color=(.6, .4, 0.1))
    # draw the feed
    reflectarray.feed.draw_feed(scale=1)
    # draw the feed lcs
    reflectarray.feed.plot_lcs(scale_factor=0.05)
    ## test the computation of the incident tangential field and its visualization
    # compute the incident tangential field
    # print computing field
    print('computing incident tangential field... ')
    Ex_l, Ey_l, Ez_l = reflectarray.compute_incident_tangential_field()
    # draw the tangential field
    print('drawing tangential field...')
    reflectarray.draw_tangential_e_field(color=(0, 1, 0), scale_factor=0.01)
    ml.show()
    # # %% recalculate the tangential field in global coordinates and plot it to compare
    # # initialize the global vector field
    # Ex_g = np.zeros_like(Ex_l)
    # Ey_g = np.zeros_like(Ey_l)
    # Ez_g = np.zeros_like(Ez_l)
    # for i in range(array.points.shape[1]):
    #     # transform the local e field in the global frame
    #     # create the transformation matrix (same of array.draw_elements_mayavi)
    #     R = np.empty((3, 3))
    #     R[:, 0] = array.el_x[:, i]
    #     R[:, 1] = array.el_y[:, i]
    #     R[:, 2] = array.el_z[:, i]
    #     # compute the global e field
    #     Ex_g[i], Ey_g[i], Ez_g[i] = R @ np.array([Ex_l[i], Ey_l[i], Ez_l[i]])
    #
    # # draw the tangential field
    # obj = ml.quiver3d(array.points[0, :], array.points[1, :], array.points[2, :], np.abs(Ex_g), np.abs(Ey_g), np.abs(Ez_g),
    #             scalars=np.angle(Ex_g), colormap='hsv', scale_mode='none', scale_factor=0.01)
    # obj.glyph.color_mode = 'color_by_scalar'
    # ml.show()