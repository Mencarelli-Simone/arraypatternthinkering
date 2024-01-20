# Simone Mencarelli
# November 2023
# This file contains a class to describe the interface for a feed antenna to be used in a reflectarray
# the class might be extended to include specific feeds or file-defined feed radiation patterns.
# %% includes
import numpy as np
from numpy import pi, sin, cos, tan, arcsin, exp
from matplotlib import pyplot as plt
from mayavi import mlab as ml
from radartools.utils import meshSph2cart
# SET PYQT5 RENDER
import matplotlib
import mayavi

matplotlib.use('Qt5Agg')


# %% Functions

# %% Feed antenna class
class FeedAntenna():
    def __init__(self, x, y, z, x_norm, y_norm, z_norm, x_tan, y_tan, z_tan, frequency, c=299792458, pol='x'):
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
        :param pol: polarization, optional 'x' or 'y', defined according to Ludwig3 convention for x and y CO-pol
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
        self.z = self.z / np.linalg.norm(self.z)
        # tangent of the feed antenna
        self.x = np.array([x_tan, y_tan, z_tan])
        self.x = self.x / np.linalg.norm(self.x)
        # y-axis of the feed antenna
        self.y = np.cross(self.z, self.x)
        # polarization
        self.pol = pol
        # graphics
        self.feed_wireframe = None
        self.set_feed_wireframe()

    def e_field(self, r, theta, phi, phase_off=False):
        """
        Compute the electric field radiated by the feed antenna at the specified point
        :param r: feed local radius
        :param theta: feed local theta
        :param phi: feed local phi
        :param pol: polarization, 'x' or 'y', defined according to Ludwig3 convention for x and y CO-pol
        :return: e theta e phi
        """
        # dummy unitary amplitude, phase dependent on r
        amplitude = 1
        phase = exp(-1j * 2 * pi * r / self.wavelength)
        if phase_off:
            phase = 1
        # phase = 1
        if self.pol == 'x':
            # x polarization, no dependence on r
            e_theta = amplitude * phase * cos(phi)
            e_phi = -amplitude * phase * sin(phi)
        else:
            # y polarization, no dependence on r
            e_theta = amplitude * phase * sin(phi)
            e_phi = amplitude * phase * cos(phi)
        return np.zeros_like(e_theta), e_theta, e_phi


    def e_field_cartesian(self, x, y, z, phase_off=False):
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
        e_r, e_theta, e_phi = self.e_field(r, theta, phi, phase_off=phase_off)
        # from the local spherical to the local cartesian
        e_x = e_r * sin(theta) * cos(phi) + e_theta * cos(theta) * cos(phi) - e_phi * sin(phi)
        e_y = e_r * sin(theta) * sin(phi) + e_theta * cos(theta) * sin(phi) + e_phi * cos(phi)
        e_z = e_r * cos(theta) - e_theta * sin(theta)
        # reshape the output vectors
        e_x = e_x.reshape(shape)
        e_y = e_y.reshape(shape)
        e_z = e_z.reshape(shape)
        return e_x, e_y, e_z

    def e_field_gcs(self, x, y, z, phase_off=False):
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
        # rotation matrix from the local to the global
        R = np.empty([3, 3])
        R[:, 0] = self.x
        R[:, 1] = self.y
        R[:, 2] = self.z
        # local cartesian coordinates
        x_lcs, y_lcs, z_lcs = R.T @ (np.array([x, y, z]) - np.repeat(self.pos.reshape(3, 1), x.shape[0], 1))  # todo debug
        # compute the electric field in the local cartesian system
        e_x, e_y, e_z = self.e_field_cartesian(x_lcs, y_lcs, z_lcs, phase_off=phase_off)
        # convert the field vectors from the local to the global cartesian system
        e_x, e_y, e_z = R @ np.array([e_x, e_y, e_z])
        # reshape the output vectors
        e_x = e_x.reshape(shape)
        e_y = e_y.reshape(shape)
        e_z = e_z.reshape(shape)
        return e_x, e_y, e_z

    # graphics
    def set_feed_wireframe(self, list_of_arrays=None):
        """
        Set the wireframe of the feed antenna
        :param list_of_arrays: optional list of arrays to draw the wireframe
        :return:
        """
        if list_of_arrays is not None:
            self.feed_wireframe = list_of_arrays
        else:
            self.feed_wireframe = []
            # square wavelength side
            a = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]]) * self.wavelength / 2
            self.feed_wireframe.append(a)
            # edges of the pyramid
            b = np.array([[0, 0, 0], [1, 1, 1]]) * self.wavelength / 2
            self.feed_wireframe.append(b)
            c = np.array([[0, 0, 0], [-1, 1, 1]]) * self.wavelength / 2
            self.feed_wireframe.append(c)
            d = np.array([[0, 0, 0], [-1, -1, 1]]) * self.wavelength / 2
            self.feed_wireframe.append(d)
            e = np.array([[0, 0, 0], [1, -1, 1]]) * self.wavelength / 2
            self.feed_wireframe.append(e)

    def draw_feed(self, scale=1, **kwargs):
        """
        draws the feed antenna in mayavi
        :return:
        """
        # create rotation matrix from the local to the global
        R = np.array([self.x, self.y, self.z]).T
        for array in self.feed_wireframe:
            # apply rotation matrix
            array = R @ array.T + self.pos.reshape(3, 1)
            ml.plot3d(array[0, :] * scale, array[1, :] * scale, array[2, :] * scale, tube_radius=None, **kwargs)

    def plot_lcs(self, **kwargs):
        """
        Plot the local coordinate system of the feed antenna
        :param kwargs: suggested scale_factor=wavelength
        :return:
        """
        # plot the local coordinate system
        ml.quiver3d(self.pos[0], self.pos[1], self.pos[2], self.x[0], self.x[1], self.x[2],
                    color=(1, 0, 0), **kwargs)
        ml.quiver3d(self.pos[0], self.pos[1], self.pos[2], self.y[0], self.y[1], self.y[2],
                    color=(0, 1, 0), **kwargs)
        ml.quiver3d(self.pos[0], self.pos[1], self.pos[2], self.z[0], self.z[1], self.z[2],
                    color=(0, 0, 1), **kwargs)


# todo extend the feed class to a cosine pattern feed

# %% Test code
# main
if __name__ == "__main__":
    # %% test the feed antenna
    # create the feed antenna
    feed = FeedAntenna(0, 0, 0, 0, 0, 1, 1, 0, 0, 10e9)
    # create the points on
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    # compute the electric field
    e_x, e_y, e_z = feed.e_field_gcs(x, y, z, phase_off=True)
    # plot the electric field
    fig, ax = plt.subplots(1)
    ax.plot(x, e_x, label="x")
    ax.plot(x, e_y, label="y")
    ax.plot(x, e_z, label="z")
    ax.legend()
    plt.show()
    # %% test the feed antenna
    # create the feed antenna
    feed = FeedAntenna(0, 0, 0, 0 , 0, 1, 1, 0, 0, 10e9)
    # create the points
    theta = np.linspace(-pi / 2, pi / 2, 100)
    phi = np.ones_like(theta) * pi / 2
    r = np.ones_like(theta)
    # in cartesian
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    # compute the electric field
    e_x, e_y, e_z = feed.e_field_cartesian(x, y, z, phase_off=True)
    # plot the electric field
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_x), label="x")
    ax.plot(theta, np.abs(e_y), label="y")
    ax.plot(theta, np.abs(e_z), label="z")
    ax.legend()
    plt.show()
    # create the points
    theta = np.linspace(-pi / 2, pi / 2, 100)
    phi = np.ones_like(theta) * pi / 10
    r = np.ones_like(theta)
    # in cartesian
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    # compute the electric field
    e_x, e_y, e_z = feed.e_field_cartesian(x, y, z, phase_off=True)
    # plot the electric field
    fig, ax = plt.subplots(1)
    ax.plot(theta, np.abs(e_x), label="x")
    ax.plot(theta, np.abs(e_y), label="y")
    ax.plot(theta, np.abs(e_z), label="z")
    ax.legend()
    plt.show()
    # %% spherical field plot for the isotropic x or y polarized antenna
    # sphere coordinates
    theta = np.linspace(0, pi, 19)
    phi = np.linspace(0, pi * 2, 37)
    # meshgrid
    theta, phi = np.meshgrid(theta, phi)
    # convert to cartesian
    r = 1
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    # calculate the electric field
    feed.pol = 'y'
    e_x, e_y, e_z = feed.e_field_cartesian(x, y, z, phase_off=True)
    # e_r, e_t, e_p = feed.e_field(np.ones_like(theta), theta, phi)
    # shape = theta.shape
    # theta = theta.reshape(-1)
    # phi = phi.reshape(-1)
    # # spherical to cartesian transformation matrix
    # S = np.array([[sin(theta) * cos(phi), cos(theta) * cos(phi), -sin(phi)],
    #               [sin(theta) * sin(phi), cos(theta) * sin(phi), cos(phi)],
    #               [cos(theta), -sin(theta), np.zeros_like(theta)]])
    # theta = theta.reshape(shape)
    # phi = phi.reshape(shape)
    # # reshape the electric field
    # shape = e_r.shape
    # e_r = e_r.reshape(-1)
    # e_t = e_t.reshape(-1)
    # e_p = e_p.reshape(-1)
    # e_x = np.zeros_like(e_r)
    # e_y = np.zeros_like(e_r)
    # e_z = np.zeros_like(e_r)
    # for i in range(e_r.shape[0]):
    #     e_x[i], e_y[i], e_z[i] = S[:, :, i] @ np.array([e_r[i], e_t[i], e_p[i]])
    # print(e_x.shape)
    # %%
    # e_x = e_x.reshape(shape)
    # e_y = e_y.reshape(shape)
    # e_z = e_z.reshape(shape)
    # plot using mayavi
    ml.figure(1)
    ml.clf()
    vf = mayavi.tools.pipeline.vector_scatter(x, y, z, e_x, e_y, e_z)
    ml.pipeline.vectors(vf, mask_points=1, scale_factor=0.08)
    ml.points3d(x, y, z, mask_points=1, scale_factor=0.01)
    # plot the gcs
    ml.quiver3d(0, 0, 0, 1, 0, 0, scale_factor=1, color=(1, 0, 0))
    ml.quiver3d(0, 0, 0, 0, 1, 0, scale_factor=1, color=(0, 1, 0))
    ml.quiver3d(0, 0, 0, 0, 0, 1, scale_factor=1, color=(0, 0, 1))


    # ml.show()

    # %% flowlines
    def flowline(theta_0, phi_0, delta_l, iterations, pol='x'):
        """

        :param theta_0:
        :param phi_0:
        :param delta_l:
        :param iterations:
        :param pol:
        :return: theta and phi coordinates
        """
        # init the flowline
        l = np.zeros([2, iterations])
        # first element
        l[:, 0] = np.array([theta_0, phi_0])
        # delta
        dl = delta_l
        # iterate
        for i in (range(1, iterations - 1)):
            phi = l[1, i - 1]
            theta = l[0, i - 1]
            # compute the new element
            if pol == 'x':
                l[:, i] = l[:, i - 1] + dl * np.array([cos(phi), -sin(phi)]) * np.array(
                    [1, 1 / sin(theta)])  # normalization for gradient
                if theta == 0:
                    l[:, i] = l[:, i - 1] + dl * np.array([cos(phi), -sin(phi)]) * np.array([1, 1])
            else:
                l[:, i] = l[:, i - 1] + dl * np.array([sin(phi), cos(phi)]) * np.array([1, 1 / sin(theta)])
                if theta == 0:
                    l[:, i] = l[:, i - 1] + dl * np.array([sin(phi), cos(phi)]) * np.array([1, 1])
            # update the delta
            dl = delta_l / np.sqrt(np.dot((l[:, i] - l[:, i - 1]), (l[:, i] - l[:, i - 1]))) * dl
            # print(dl, delta_l / np.sqrt(np.dot((l[:, i] - l[:, i - 1]),(l[:, i] - l[:, i - 1]))), delta_l)
            # if nan
            if np.isnan(dl).any():
                dl = delta_l

        return l[0, :], l[1, :]


    # starting point

    tt = np.linspace(-pi, pi, 19)
    # tt = np.linspace(0, pi/2, 10)
    pp = np.ones_like(tt) * 0
    # mayavi plot
    ml.figure(1)
    # ml.clf()
    for i in range(tt.shape[0]):
        t = tt[i]
        p = pp[i]
        # compute the flowline
        th, ph = flowline(t, p, pi / 8000, 30000, pol='y')
        # convert to cartesian
        r = 1
        x = r * sin(th) * cos(ph)
        y = r * sin(th) * sin(ph)
        z = r * cos(th)

        nodes = ml.plot3d(x[0:-2], y[0:-2], z[0:-2], tube_radius=None)
        # nodes.glyph.scale_mode = 'scale_by_vector'
        colors = i * np.ones_like(x)
        nodes.mlab_source.dataset.point_data.scalars = colors / tt.shape[0]
    ml.show()

    # the field lines are just circles

    # %% obliquity factor analysis
    theta = np.linspace(-pi / 2, pi / 2, 19)
    of = (1 + cos(theta)) / 2
    # circular coordinates plot
    z = of * cos(theta)
    x = of * sin(theta)
    # parametric plot
    fig, ax = plt.subplots(1)
    ax.plot(x, z)

    # %% test the feed antenna drawing
    feed.draw_feed(scale=10)
    ml.show()

    #%% Test the feed antenna in the global cartesian system
    # create the feed antenna
    feed = FeedAntenna(0, 0, 0, 1, 0, 0, 1, 0, 0, 10e9)
    # create the points
    # sphere coordinates
    theta = np.linspace(0, pi, 19)
    phi = np.linspace(0, pi * 2, 37)
    # meshgrid
    theta, phi = np.meshgrid(theta, phi)
    r = np.ones_like(theta)
    # in cartesian
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    # from lcs to gcs
    R = np.empty([3, 3])
    R[:, 0] = feed.x # the lcs vectors appear as columns
    R[:, 1] = feed.y
    R[:, 2] = feed.z
    x, y, z = R @ np.array([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    # compute the electric field
    e_x, e_y, e_z = feed.e_field_gcs(x, y, z, phase_off=True)
    # plot the electric field quivers mayavi
    ml.figure(2)
    ml.clf()
    ml.quiver3d(x, y, z, e_x, e_y, e_z, mask_points=1, scale_factor=0.1)
    # plot the feed antenna axes
    feed.plot_lcs(scale_factor=0.1)
    # plot the feed antenna
    feed.draw_feed(scale=0.3)
    ml.show() # PASS