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
        :param phase_shift: phase shift of the element  single value or [N] 1-d!
        :return: scattering matrix [N,2,2]
        """
        # for the ideal element the scattering matrix is a diagonal matrix with the phase shift
        # on the diagonal corresponding to the input parameter
        # if phase shift is an index
        if type(phase_shift) is int:
            length = 1
        else:
            length = len(phase_shift)
        gamma = np.zeros([length, 2, 2], dtype=complex)
        gamma[:, 0, 0] = np.exp(1j * phase_shift)
        gamma[:, 1, 1] = np.exp(1j * phase_shift)
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
        #  reflected fields lcs
        self.Ex_r = None
        self.Ey_r = None
        self.Ez_r = None
        # also the H field of the reflected plane wave
        self.Hx_r = None
        self.Hy_r = None
        self.Hz_r = None
        # also the k vector of the reflected plane wave
        self.kx = None
        self.ky = None
        self.kz = None
        # incidence angle for each element (in LCS of the elements)
        self.theta_inc = None
        self.phi_inc = None
        self.r = None  # distance from the feed to the element
        # phase shifts of the elements
        self.phase_shift = None
        # collimation direction
        self.theta_broadside = None
        self.phi_broadside = None
        self.__update__(collimate=False, reflect=False)

    def __update__(self, collimate=False, reflect=True):
        """
        Update the reflectarray model
        :return:
        """
        print('updating reflectarray model...')
        # compute the incident tangential field
        self.compute_incident_tangential_field()
        # collimate the beam
        if collimate:
            if self.theta_broadside is None:
                self.collimate_beam(0, 0)
            else:
                self.collimate_beam(self.theta_broadside, self.phi_broadside)
        # compute the elements angle of incidence
        self.compute_elements_angle_of_incidence()
        # compute the reflected tangential field
        if reflect:
            self.compute_reflected_tangential_field()

    def compute_incident_tangential_field(self, phase_off=False):
        # This method shall be jitted to run in parallel
        # 1. get global x, y, z e field for each element in the array
        Ex, Ey, Ez = self.feed.e_field_gcs(self.array.points[0], self.array.points[1], self.array.points[2],
                                           phase_off=phase_off)
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
        r = np.zeros(self.array.points.shape[1])
        # initialize feed pos lcs
        feed_pos_lcs = np.zeros_like(self.array.points)
        # compute the angles
        for i in range(self.array.points.shape[1]):
            # convert the feed position in the lcs of the element
            # create the transformation matrix
            R = np.empty((3, 3))
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            # compute the feed position in the lcs of the element
            feed_pos_lcs[:, i] = R.T @ (self.feed.pos - self.array.points[:, i])
        # convert to spherical coordinates
        r = np.linalg.norm(feed_pos_lcs, axis=0)
        theta_inc = np.where(r != 0, np.arccos(feed_pos_lcs[2, :] / r), 0)
        phi_inc = np.where(r != 0, np.arctan2(feed_pos_lcs[1, :], feed_pos_lcs[0, :]), 0)
        self.theta_inc = theta_inc
        self.phi_inc = phi_inc
        self.r = r
        return theta_inc, phi_inc, r

    def collimate_beam(self, theta_broadside, phi_broadside, polarization='x', phase_shift_offset=0):
        """
        Compute the phase shifts required to collimate the beam in the direction of theta_broadside, phi_broadside
        :param theta_broadside:
        :param phi_broadside:
        :param polarization: component of the elemental incident field to be collimated
        :param phase_shift_offset: optional, fixed phase shift offset to be added to the phase shifts 0-2pi
        :return:
        """
        print('collimating beam...')
        # get the electric field phase for each element desired polarization
        phase = np.angle(self.Ex_i) if polarization == 'x' else np.angle(self.Ey_i)
        # compute the desired aperture phase shift
        # 1: find the reference plane for the incident field(theta phi) and make it intersect with the origin
        # 1.1: find the normal to the plane from theta phi
        norm = np.array([sin(theta_broadside) * cos(phi_broadside), sin(theta_broadside) * sin(phi_broadside),
                         cos(theta_broadside)])
        # 2: find the distance between the plane and each element using the matrix product point coordinate- norm
        d = self.array.points.T @ norm
        # 3: compute the phase shift for the collimated beam + - upchirp down chirp shapes
        desired_phase = (-2 * pi * d / self.feed.wavelength)
        # 4: required phase shift offset
        required_phase_shift = (desired_phase - phase + phase_shift_offset) % (2 * pi) # this is correct
        # returns the required phase shifts to collimate the beam
        self.phase_shift = required_phase_shift
        # save the collimation direction
        self.theta_broadside = theta_broadside
        self.phi_broadside = phi_broadside
        return required_phase_shift

    def compute_reflected_tangential_field(self):
        """"
        Call the methods of the cell model to compute the reflected tangential field
        has to be called after compute_incident_tangential_field and compute_elements_angle_of_incidence
        these are to be utilised directly to find the reflected field
        :return: Erx, Ery: reflected tangential field LCS of the elements

        """
        # compute the scattering matrix
        gamma = self.cell.scattering_matrix(self.theta_inc, self.phi_inc, self.phase_shift)
        # shape correctly the incident field vectors [N,2] (lcs of the elements)
        Ei = np.vstack((self.Ex_l, self.Ey_l)).T
        # compute the reflected field, matrix stacks multiplications
        Er = gamma @ Ei.reshape(-1, 2, 1)
        self.Ex_r = Er[:, 0].reshape(-1)  # this can make a mess if the array is not linear
        self.Ey_r = Er[:, 1].reshape(-1)
        # the plane wave reflection direction is given by theta and phi + pi/2
        # from spherical to cartesian
        kx = sin(self.theta_inc) * cos(self.phi_inc + pi)
        ky = sin(self.theta_inc) * sin(self.phi_inc + pi)
        kz = cos(self.theta_inc)
        self.kx = kx
        self.ky = ky
        self.kz = kz
        # the z component of e can be calculated (eq 8 in Prado et all, Efficient Crosspolar optimization of shaped-beam dual polarized reflectarrays)
        self.Ez_r = (- kx * self.Ex_r - ky * self.Ey_r) / kz
        # compute the H field as (k x E) / eta (cross product)
        self.Hx_r = (ky * self.Ez_r - kz * self.Ey_r) / self.array.element_antenna.eta
        self.Hy_r = (kz * self.Ex_r - kx * self.Ez_r) / self.array.element_antenna.eta
        self.Hz_r = (kx * self.Ey_r - ky * self.Ex_r) / self.array.element_antenna.eta
        # with this we already have the tangetial field components (in lcs) for the aperture integration.
        return self.Ex_r, self.Ey_r, self.Ez_r, self.Hx_r, self.Hy_r, self.Hz_r

    def far_field_huygens(self, theta, phi):
        """
        Compute the far field of the reflectarray
        has to be called after compute_reflected_tangential_field
        :param theta: polar angle
        :param phi: azimuthal angle
        :return: far field E_theta, E_phi
        """
        print('computing far field...')  # todo change this to an aperture integration, but keep this for testing
        # x component
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Ex_r
        # set the elements polarisation to x
        self.array.polarization = "x"
        # set field type
        self.array.source_type = "huygens"
        # 2. compute the far field
        Etheta_x, Ephi_x = self.array.far_field(theta, phi)
        # y component
        # set the elements polarisation to y
        self.array.polarization = "y"
        # set field type
        self.array.source_type = "huygens"
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Ey_r
        # 2. compute the far field
        Etheta_y, Ephi_y = self.array.far_field(theta, phi)
        # sum the components
        Etheta = Etheta_x + Etheta_y
        Ephi = Ephi_x + Ephi_y
        return Etheta, Ephi

    def far_field(self, theta, phi):
        """
        Compute the far field of the reflectarray
        has to be called after compute_reflected_tangential_field
        :param theta: polar angle
        :param phi: azimuthal angle
        :return: far field E_theta, E_phi
        """
        print('computing far field...')  # todo change this to an aperture integration, but keep this for testing
        # all0cate the output
        Etheta_e_x = 1j * np.zeros_like(theta)
        Ephi_e_x = 1j * np.zeros_like(theta)
        Etheta_e_y = 1j * np.zeros_like(theta)
        Ephi_e_y = 1j * np.zeros_like(theta)
        Etheta_m_x = 1j * np.zeros_like(theta)
        Ephi_m_x = 1j * np.zeros_like(theta)
        Etheta_m_y = 1j * np.zeros_like(theta)
        Ephi_m_y = 1j * np.zeros_like(theta)
        # x component electric
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Ex_r
        # set the elements polarisation to x
        self.array.polarization = "x"
        self.array.source_type = "electric"
        # 2. compute the far field
        Etheta_e_x, Ephi_e_x = self.array.far_field(theta, phi)
        # y component electric
        # set the elements polarisation to y
        self.array.polarization = "y"
        self.array.source_type = "electric"
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Ey_r
        # 2. compute the far field
        Etheta_e_y, Ephi_e_y = self.array.far_field(theta, phi)
        # x component magnetic
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Hx_r
        # set the elements polarisation to x
        self.array.polarization = "x"
        self.array.source_type = "magnetic"
        # 2. compute the far field
        Etheta_m_x, Ephi_m_x = self.array.far_field(theta, phi)
        # y component magnetic
        # set the elements polarisation to y
        self.array.polarization = "y"
        self.array.source_type = "magnetic"
        # 1. set the complex excitation of the elements as the reflected field
        self.array.excitations = self.Hy_r
        # 2. compute the far field
        Etheta_m_y, Ephi_m_y = self.array.far_field(theta, phi)
        # sum the components
        Etheta = Etheta_e_x + Etheta_e_y + Etheta_m_x + Etheta_m_y
        Ephi = Ephi_e_x + Ephi_e_y + Ephi_m_x + Ephi_m_y
        return Etheta, Ephi

    def co_cross_pol(self, theta, phi, polarization='x', E_theta=None, E_phi=None):
        """
        extracts the co and cross polarized components of the far field using Ludwig3 definition
        :param theta: theta angle
        :param phi: phi angle same shape of theta
        :param polarization: polarization of the co polarized component x or y (default x)
        :return:
        """
        shape = theta.shape
        theta = theta.reshape(-1)
        phi = phi.reshape(-1)
        # compute the far field
        if E_theta is None or E_phi is None:
            E_theta, E_phi = self.far_field(theta, phi)
        # compute the co and cross polarized components
        if polarization == 'x':
            # x co x pol matrix Ludwig3 eq 4.34 [1]
            M = np.zeros((theta.shape[0], 2, 2))
            M[:, 0, 0] = cos(phi)
            M[:, 0, 1] = -sin(phi)
            M[:, 1, 0] = - sin(phi)
            M[:, 1, 1] = - cos(phi)
            b = M @ np.vstack([E_theta, E_phi]).T.reshape(-1, 2, 1)
            E_co = b[:, 0, 0]
            E_cross = b[:, 1, 0]

        elif polarization == 'y':
            # y co x pol matrix Ludwig3 eq 4.35 [1]
            M = np.zeros((theta.shape[0], 2, 2))
            M[:, 0, 0] = sin(phi)
            M[:, 0, 1] = cos(phi)
            M[:, 1, 0] = cos(phi)
            M[:, 1, 1] = - sin(phi)
            b = M @ np.vstack([E_theta, E_phi]).T.reshape(-1, 2, 1)
            E_co = b[:, 0, 0]
            E_cross = b[:, 1, 0]
        else:
            raise ValueError('polarization must be x or y')
        # reshape the output
        E_co = E_co.reshape(shape)
        E_cross = E_cross.reshape(shape)
        return E_co, E_cross

    def radiated_power_per_element(self):
        # E x H* / 2
        # poynting vector
        Sx = (self.Ey_r * np.conj(self.Hz_r) - self.Ez_r * np.conj(self.Hy_r)) / 2
        Sy = (self.Ez_r * np.conj(self.Hx_r) - self.Ex_r * np.conj(self.Hz_r)) / 2
        Sz = (self.Ex_r * np.conj(self.Hy_r) - self.Ey_r * np.conj(self.Hx_r)) / 2
        # power per element, ie S dot z * surface area
        P = np.abs(Sz * self.array.element_antenna.L * self.array.element_antenna.W)
        return P

    def directive_gain_huygens(self, theta, phi, polarization='x', E_theta=None, E_phi=None):
        """
        Compute the directive gain of the reflectarray
        :param theta: polar angle
        :param phi: azimuthal angle
        :param polarization: polarization of the co polarized component x or y (default x)
        :return: directive gain
        """
        # compute the far field
        if E_theta is None or E_phi is None:
            E_theta, E_phi = self.far_field_huygens(theta, phi)
        E_theta, E_phi = self.far_field_huygens(theta, phi)
        # compute the co and cross polarized components
        E_co, E_cross = self.co_cross_pol(theta, phi, polarization, E_theta, E_phi)
        # compute the directive gains (co and cross) dividing the radiated power by the
        # total power flux on the array elements (unpolarised) Ludwig3 eq 4.36 [1]
        # todo change this to a proper flux with poynting vector and all but keep this version for testing
        radiated_power = (
                np.sum(np.abs(self.Ex_r) ** 2 + np.abs(self.Ey_r) ** 2) / (2 * self.array.element_antenna.eta) *
                self.array.element_antenna.L * self.array.element_antenna.W)

        Gco = 2 * pi * np.abs(E_co) ** 2 / (self.array.element_antenna.eta * radiated_power)
        Gcross = 2 * pi * np.abs(E_cross) ** 2 / (self.array.element_antenna.eta * radiated_power)
        return Gco, Gcross

    def directive_gain(self, theta, phi, polarization='x', E_theta=None, E_phi=None):
        """
        Compute the directive gain of the reflectarray
        :param theta: polar angle
        :param phi: azimuthal angle
        :param polarization: polarization of the co polarized component x or y (default x)
        :return: directive gain
        """
        # compute the far field
        if E_theta is None or E_phi is None:
            E_theta, E_phi = self.far_field(theta, phi)
        E_theta, E_phi = self.far_field(theta, phi)
        # compute the co and cross polarized components
        E_co, E_cross = self.co_cross_pol(theta, phi, polarization, E_theta, E_phi)
        # compute the directive gains (co and cross) dividing the radiated power by the
        # total power flux on the array elements (unpolarised) Ludwig3 eq 4.36 [1]
        radiated_power = np.sum(self.radiated_power_per_element())
        Gco = 2 * pi * np.abs(E_co) ** 2 / (self.array.element_antenna.eta * radiated_power)
        Gcross = 2 * pi * np.abs(E_cross) ** 2 / (self.array.element_antenna.eta * radiated_power)
        return Gco, Gcross

    # graphics
    def draw_reflectarray(self, **kwargs):
        # draw the array
        self.array.draw_elements_mayavi(**kwargs)
        # draw the feed
        self.feed.draw_feed(scale=1)

    def draw_tangential_e_field(self, phase_color=False, **kwargs):
        """
        Draw the tangential electric field on the reflectarray surface for every element
        :param kwargs: mayavi quiver3d kwargs
        :return:
        """
        # recomputes the incident tangential field using gcs method of feed, with phase off option
        Ex_i, Ey_i, Ez_i = self.feed.e_field_gcs(self.array.points[0], self.array.points[1], self.array.points[2],
                                                 phase_off=True)
        if phase_color == False:
            # draw the e field with mayavi for every point
            ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], Ex_i,
                        Ey_i, Ez_i, **kwargs)
        else:
            obj = ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], Ex_i,
                              Ey_i, Ez_i, scalars=np.angle(self.Ex_i), scale_mode='none', **kwargs)
            obj.glyph.color_mode = 'color_by_scalar'

    def draw_specular_reflection_vector(self, **kwargs):
        # transform k in gcs
        kx_g = np.zeros_like(self.kx)
        ky_g = np.zeros_like(self.ky)
        kz_g = np.zeros_like(self.kz)
        for i in range(self.array.points.shape[1]):
            # transform the local e field in the global frame
            # create the transformation matrix (same of array.draw_elements_mayavi)
            R = np.empty((3, 3))
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            # compute the global e field
            kx_g[i], ky_g[i], kz_g[i] = R @ np.array([self.kx[i], self.ky[i], self.kz[i]])
        # draw the specular reflection vector
        ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], kx_g,
                    ky_g, kz_g, color=(0, 0, 1), **kwargs)

    def draw_reflected_field(self, **kwargs):
        # transform E_r and H_r in gcs
        Ex_g = np.zeros_like(self.Ex_r)
        Ey_g = np.zeros_like(self.Ey_r)
        Ez_g = np.zeros_like(self.Ez_r)
        Hx_g = np.zeros_like(self.Hx_r)
        Hy_g = np.zeros_like(self.Hy_r)
        Hz_g = np.zeros_like(self.Hz_r)
        # compute the phase shift for each reflected local plane wave
        free_space = np.angle(self.Ex_i)
        phase_shift = self.phase_shift + free_space
        # remove the phase shift offset from E
        Ex_r = self.Ex_r * np.exp(-1j * phase_shift)
        Ey_r = self.Ey_r * np.exp(-1j * phase_shift)
        Ez_r = self.Ez_r * np.exp(-1j * phase_shift)
        # remove the phase shift offset from H
        Hx_r = self.Hx_r * np.exp(-1j * phase_shift)
        Hy_r = self.Hy_r * np.exp(-1j * phase_shift)
        Hz_r = self.Hz_r * np.exp(-1j * phase_shift)
        for i in range(self.array.points.shape[1]):
            # transform the local e field in the global frame
            # create the transformation matrix (same of array.draw_elements_mayavi)
            R = np.empty((3, 3)).astype(complex)
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            # compute the global e field
            Ex_g[i], Ey_g[i], Ez_g[i] = R @ np.array([Ex_r[i], Ey_r[i], Ez_r[i]])
            Hx_g[i], Hy_g[i], Hz_g[i] = R @ np.array([Hx_r[i], Hy_r[i], Hz_r[i]])
        # draw the specular reflection vector

        # print the maximum and minimum imaginary part of the reflected field
        print('max and min imaginary part of the reflected field:')
        print('Ex: ', np.max(np.imag(Ex_g)), np.min(np.imag(Ex_g)))
        print('Ey: ', np.max(np.imag(Ey_g)), np.min(np.imag(Ey_g)))
        print('Ez: ', np.max(np.imag(Ez_g)), np.min(np.imag(Ez_g)))
        print('Hx: ', np.max(np.imag(Hx_g)), np.min(np.imag(Hx_g)))
        print('Hy: ', np.max(np.imag(Hy_g)), np.min(np.imag(Hy_g)))
        print('Hz: ', np.max(np.imag(Hz_g)), np.min(np.imag(Hz_g)))
        # print maximum and minimum real part of the reflected field
        print('max and min real part of the reflected field:')
        print('Ex: ', np.max(np.real(Ex_g)), np.min(np.real(Ex_g)))
        print('Ey: ', np.max(np.real(Ey_g)), np.min(np.real(Ey_g)))
        print('Ez: ', np.max(np.real(Ez_g)), np.min(np.real(Ez_g)))
        print('Hx: ', np.max(np.real(Hx_g)), np.min(np.real(Hx_g)))
        print('Hy: ', np.max(np.real(Hy_g)), np.min(np.real(Hy_g)))
        print('Hz: ', np.max(np.real(Hz_g)), np.min(np.real(Hz_g)))
        # draw the specular reflection vector
        ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], np.real(Ex_g),
                    np.real(Ey_g), np.real(Ez_g),
                    color=(1, 0, 0), **kwargs)
        ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :],
                    np.real(Hx_g) * self.array.element_antenna.eta,
                    np.real(Hy_g) * self.array.element_antenna.eta,
                    np.real(Hz_g) * self.array.element_antenna.eta,
                    color=(0, 1, 0), **kwargs)

    def draw_reflected_tangential_e_field(self, phase_color=False, **kwargs):
        """
        Draw the tangential electric field on the reflectarray surface for every element
        :param kwargs: mayavi quiver3d kwargs
        :return:
        """
        self.__update__()
        # recomputes the incident tangential field using gcs method of feed, with phase off option
        Ex_i, Ey_i, Ez_i = self.Ex_r, self.Ey_r, np.zeros_like(self.Ex_r)
        # convert to gcs
        for i in range(self.array.points.shape[1]):
            # transform the local e field in the global frame
            # create the transformation matrix (same of array.draw_elements_mayavi)
            R = np.empty((3, 3))
            R[:, 0] = self.array.el_x[:, i]
            R[:, 1] = self.array.el_y[:, i]
            R[:, 2] = self.array.el_z[:, i]
            # compute the global e field
            Ex_i[i], Ey_i[i], Ez_i[i] = R @ np.array([Ex_i[i], Ey_i[i], Ez_i[i]])
            # remove the angle
            Ex_i[i], Ey_i[i], Ez_i[i] = np.abs(Ex_i[i]), np.abs(Ey_i[i]), np.abs(Ez_i[i])

        if phase_color == False:
            # draw the e field with mayavi for every point
            ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :], np.real(Ex_i),
                        np.real(Ey_i), np.real(Ez_i), **kwargs)
        else:
            obj = ml.quiver3d(self.array.points[0, :], self.array.points[1, :], self.array.points[2, :],
                              np.real(Ex_i).reshape(-1),
                              np.real(Ey_i).reshape(-1), np.real(Ez_i).reshape(-1),
                              scalars=np.angle(self.Ex_i).reshape(-1), scale_mode='none', **kwargs)
            obj.glyph.color_mode = 'color_by_scalar'


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
    tc = 100 * pi / 180
    ## create the array
    array, radius = create_cylindrical_array(L, W, tc, dx, freq)
    ## create the feed
    # position the feed at the focus of the cylinder on the z axis pointing down
    x = 0
    y = 0
    z = 0.5  # m
    feed = FeedAntenna(x, y, z, 0, 0, -1, 1, 0, 0, freq)
    ## create ra cell
    cell = RACell()
    # create the reflectarray
    reflectarray = ReflectArray(cell, feed, array)
    reflectarray.__update__(collimate=True, reflect=True)
    # display the RA in mayavi
    ml.figure(1, bgcolor=(0, 0, 0))
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
    reflectarray.draw_tangential_e_field(phase_color=False, color=(0, 1, 0), scale_factor=dx)
    ml.show()
    # %% recalculate the tangential field in global coordinates and plot it to compare
    # initialize the global vector field
    Ex_l, Ey_l, Ez_l = reflectarray.compute_incident_tangential_field(phase_off=True)
    Ex_g = np.zeros_like(Ex_l)
    Ey_g = np.zeros_like(Ey_l)
    Ez_g = np.zeros_like(Ez_l)
    for i in range(array.points.shape[1]):
        # transform the local e field in the global frame
        # create the transformation matrix (same of array.draw_elements_mayavi)
        R = np.empty((3, 3))
        R[:, 0] = array.el_x[:, i]
        R[:, 1] = array.el_y[:, i]
        R[:, 2] = array.el_z[:, i]
        # compute the global e field
        Ex_g[i], Ey_g[i], Ez_g[i] = R @ np.array([Ex_l[i], Ey_l[i], Ez_l[i]])

    Ex_l, Ey_l, Ez_l = reflectarray.compute_incident_tangential_field(phase_off=False)
    # draw the tangential field
    obj = ml.quiver3d(array.points[0, :], array.points[1, :], array.points[2, :], Ex_g, Ey_g,
                      Ez_g,
                      scalars=np.angle(reflectarray.Ex_i), colormap='hsv', scale_mode='none', scale_factor=dx,
                      mode='arrow')
    obj.glyph.color_mode = 'color_by_scalar'

    # %%
    print('drawing reflected tangential field...')
    reflectarray.draw_reflected_tangential_e_field(phase_color=True, scale_factor=dx)

    ml.show()

    # %%  draw the reflected plane wave vectors, they should form a right handed base e h k
    reflectarray.__update__(reflect=True)
    ml.figure(10, bgcolor=(0, 0, 0))
    ml.clf()
    # draw the array
    reflectarray.array.draw_elements_mayavi(color=(.6, .4, 0.1))
    # draw the feed
    reflectarray.feed.draw_feed(scale=1)
    # draw the feed lcs
    reflectarray.feed.plot_lcs(scale_factor=0.05)
    # draw the specular reflection vector
    reflectarray.draw_specular_reflection_vector(scale_factor=dx, mode='arrow')
    # draw the reflected field
    reflectarray.draw_reflected_field(scale_factor=dx, mode='arrow')
    ml.show() # so far all good

    # %% RA test COLLIMATION AND reflected tangential field computation and visualization
    # call the update function
    # reflectarray.__update__()
    # # redraw the reflectarray mayavi
    # scene = ml.figure(2, bgcolor=(0, 0, 0))
    # ml.clf()
    # # draw the array
    # reflectarray.array.draw_elements_mayavi(color=(.6, .4, 0.1))
    # # draw the feed
    # reflectarray.feed.draw_feed(scale=1)
    # # draw the feed lcs
    # reflectarray.feed.plot_lcs(scale_factor=0.05)
    # # draw the surface phase shifts
    # reflectarray.array.draw_element_surfaces_mayavi(parameter=reflectarray.phase_shift / (2 * pi))
    # # draw the reflected tangential field
    # ml.show()

    ################################################

    # %% test the far field
    # phi = 0 cut (azimuthal)
    theta = np.linspace(-pi / 2, pi / 2, 3600)
    phi = np.ones_like(theta) * 0
    Etheta, Ephi = reflectarray.far_field(theta, phi)
    # plot the far field
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta)), label='Etheta e-cut (non huygens)')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi)), label='Ephi e-cut (non huygens)')
    ax.set_xlabel('theta [deg]')
    ax.set_ylabel('E [dB]')
    ax.set_title('Far field')
    # add to the plot the individual x and y components
    # 1. set the complex excitation of the elements as the reflected field
    reflectarray.array.excitations = reflectarray.Ex_r
    # set the elements polarisation to x
    reflectarray.array.polarization = "x"
    # set huygens source type
    reflectarray.array.source_type = "huygens"
    # 2. compute the far field
    Etheta_x, Ephi_x = reflectarray.array.far_field(theta, phi)
    # y component
    # set the elements polarisation to y
    reflectarray.array.polarization = "y"
    # 1. set the complex excitation of the elements as the reflected field
    reflectarray.array.excitations = reflectarray.Ey_r
    # 2. compute the far field
    Etheta_y, Ephi_y = reflectarray.array.far_field(theta, phi)
    # 3. plot the components
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta_x)), label='Etheta x')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi_x)), label='Ephi x')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta_y)), label='Etheta y')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi_y)), label='Ephi y')
    ax.legend()
    plt.show()
    # phi =90 cut (polar)
    phi = np.ones_like(theta) * pi / 2
    Etheta, Ephi = reflectarray.far_field(theta, phi)
    # plot the far field
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta)), '-.', label='Etheta h-cut (non huygens)')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi)), '-.', label='Ephi h-cut (non huygens)')
    ax.legend()
    plt.show()

    # add to the plot the individual x and y components
    # 1. set the complex excitation of the elements as the reflected field
    reflectarray.array.excitations = reflectarray.Ex_r
    # set the elements polarisation to x
    reflectarray.array.polarization = "x"
    # 2. compute the far field
    Etheta_x, Ephi_x = reflectarray.array.far_field(theta, phi)
    # y component
    # set the elements polarisation to y
    reflectarray.array.polarization = "y"
    # 1. set the complex excitation of the elements as the reflected field
    reflectarray.array.excitations = reflectarray.Ey_r
    # 2. compute the far field
    Etheta_y, Ephi_y = reflectarray.array.far_field(theta, phi)
    # 3. plot the components
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta_x)), '--', label='Etheta x')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi_x)), '--', label='Ephi x')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Etheta_y)), '--', label='Etheta y')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ephi_y)), '--', label='Ephi y')
    ax.legend()
    plt.show()
    ###########################################################################################################################
    # %% test the co and cross polarized components

    theta = np.linspace(-pi / 2, pi / 2, 1800)
    phi = np.ones_like(theta) * 0

    Etheta, Ephi = reflectarray.far_field(theta, phi)
    Ex_co, Ex_cross = reflectarray.co_cross_pol(theta, phi, polarization='x', E_theta=Etheta, E_phi=Ephi)
    Ey_co, Ey_cross = reflectarray.co_cross_pol(theta, phi, polarization='y', E_theta=Etheta, E_phi=Ephi)

    # compare with old huygens method
    Etheta_huygens, Ephi_huygens = reflectarray.far_field_huygens(theta, phi)
    Ex_co_huygens, Ex_cross_huygens = reflectarray.co_cross_pol(theta, phi, polarization='x', E_theta=Etheta_huygens,
                                                                E_phi=Ephi_huygens)
    # plot the far field
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_co)), label='Ex co')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ey_co)), label='Ey co')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_cross)), '--', label='Ex cross')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ey_cross)), '--', label='Ey cross')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_co_huygens)), label='Ex co huygens')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_cross_huygens)), '--', label='Ex cross huygens')

    ax.set_xlabel('theta [deg]')
    ax.set_ylabel('E [dB]')
    ax.set_title('Far field')
    ax.legend()

    # phi = 0 cut (azimuthal)
    theta = np.linspace(-pi / 2, pi / 2, 1800)
    phi = np.ones_like(theta) * pi / 2

    Etheta, Ephi = reflectarray.far_field(theta, phi)
    Ex_co, Ex_cross = reflectarray.co_cross_pol(theta, phi, polarization='x', E_theta=Etheta, E_phi=Ephi)
    Ey_co, Ey_cross = reflectarray.co_cross_pol(theta, phi, polarization='y', E_theta=Etheta, E_phi=Ephi)
    # compare with old huygens method
    print("comparison with huygens method")
    Etheta_huygens, Ephi_huygens = reflectarray.far_field_huygens(theta, phi)
    Ex_co_huygens, Ex_cross_huygens = reflectarray.co_cross_pol(theta, phi, polarization='x', E_theta=Etheta_huygens,
                                                                E_phi=Ephi_huygens)
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_co)), label='Ex co')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ey_co)), label='Ey co')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_cross)), '--', label='Ex cross')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ey_cross)), '--', label='Ey cross')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_co_huygens)), label='Ex co huygens')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ex_cross_huygens)), '--', label='Ex cross huygens')
    ax.legend()
    plt.show()
    ## aperture comparison
    from radartools.farField import UniformAperture

    print("comparison with uniform aperture")
    # aperture
    eqap = UniformAperture(2.0250, 0.3021, freq)
    # theta and phi axes, phi = 0 cut
    theta = np.linspace(-pi / 2, pi / 2, 1800)
    phi0 = np.ones_like(theta) * 0
    phi90 = np.ones_like(theta) * pi / 2
    Et, Ep = eqap.mesh_E_field_theor(theta, phi0)
    Et1, Ep1 = eqap.mesh_E_field_theor(theta, phi90)
    # plot the directive gain
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Ep)), label='Ep phi0 eqap')
    ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(Et1)), label='Ep phi90 eqap')
    ax.legend()
    plt.show()

    # # %% power on surface visualization for the x component
    # # compute the power radiated on the surface
    # Prad_x = np.abs(reflectarray.Ex_r) ** 2 / (2 * reflectarray.array.element_antenna.eta)
    # Prad = (np.abs(reflectarray.Ex_r) ** 2 + np.abs(reflectarray.Ey_r) ** 2) / (
    #         2 * reflectarray.array.element_antenna.eta)
    # # plot the power on the surface
    # # create mayavi figure
    # ml.figure(3, bgcolor=(0, 0, 0))
    # ml.clf()
    # # draw the ra
    # reflectarray.draw_reflectarray()
    # # draw the power on the surface
    # reflectarray.array.draw_element_surfaces_mayavi(parameter=Prad_x / np.max(Prad))
    # ml.show()
    # %% power on surface visualization for the y component
    # compute the power radiated on the surface
    Prad_y = np.abs(reflectarray.Ey_r) ** 2 / (2 * reflectarray.array.element_antenna.eta)
    # plot the power on the surface
    # create mayavi figure
    ml.figure(4, bgcolor=(0, 0, 0))
    ml.clf()
    # draw the ra
    reflectarray.draw_reflectarray()
    # draw the power on the surface
    reflectarray.array.draw_element_surfaces_mayavi(parameter=Prad_y / np.max(Prad_y))
    ml.show()

    # %% test the directive gain and compare plots with uniform aperture gain,
    # theta and phi axes, phi = 0 cut
    theta = np.linspace(-pi / 2, pi / 2, 1800)
    phi0 = np.ones_like(theta) * 0
    phi90 = np.ones_like(theta) * pi / 2
    # compute the directive gain
    Gco, Gcross = reflectarray.directive_gain_huygens(theta, phi0, polarization='x')
    # plot the directive gain
    fig, ax = plt.subplots(1)
    ax.plot(theta * 180 / pi, 10 * np.log10(Gco), label='Gco phi0 huygens')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gcross), label='Gcross phi0 huygens')
    Gco, Gcross = reflectarray.directive_gain_huygens(theta, phi90, polarization='x')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gco), label='Gco phi90 huygens')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gcross), label='Gcross phi90 huygens')
    Gco, Gcross = reflectarray.directive_gain(theta, phi0, polarization='x')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gco), label='Gco phi0 (full)')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gcross), label='Gcross phi0 (full)')
    Gco, Gcross = reflectarray.directive_gain(theta, phi90, polarization='x')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gco), label='Gco phi90 (full)')
    ax.plot(theta * 180 / pi, 10 * np.log10(Gcross), label='Gcross phi90 (full)')
    ax.set_xlabel('theta [deg]')
    ax.set_ylabel('G [dB]')
    ax.set_title('Directive gain')
    ax.legend()
    plt.show()

    # %% compute the gain for an equivalent aperture
    from radartools.farField import UniformAperture

    # aperture
    eqap = UniformAperture(2.0250, 0.3021, freq)
    # compute the gain
    G = eqap.mesh_gain_pattern_theor(theta, phi0)
    G1 = eqap.mesh_gain_pattern_theor(theta, phi90)
    # plot the directive gain
    ax.plot(theta * 180 / pi, 10 * np.log10(G), label='Gco phi0 eqap')
    ax.plot(theta * 180 / pi, 10 * np.log10(G1), label='Gco phi90 eqap')
    ax.legend()
    plt.show()
