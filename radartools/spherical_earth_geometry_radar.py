# This file contains all the geometrical transformation
# functions and a modified version of the class radar geometry
# Doppler to azimuth transforms are also included ##

from numpy import cos, sin
import numpy as np
from numba import jit, prange
from scipy import integrate
from radartools.farField import UniformAperture
from radartools.design_functions import pd_from_nesz_res


## Geometry Class simplified for spherical geometry ##
# The GCS is positioned on the spherical earth ground nadir point below the radar at time t = 0 #
class RadarGeometry():
    """
    describes the rotation, speed and position of the radar with respect to a global reference system
    provides methods to retrieve radar position in time and taget related delays
    """

    def __int__(self):
        self.trajectory_angle = 0  # store it to be used in the speed vector
        self.side_looking_angle = 0  # looking angle from the down looking direction to the right of the azimuth path
        self.forward_squint_angle = 0  # forward looking angle toward the walk path

    def set_rotation(self, looking_angle: float, trajectory_angle: float, squint_angle: float):
        """
        sets the rotation of the satellite antenna and creates a reference system with axes accordingly rotated
        this sets also the velocity accordingly to. all angles are in radiant
        :param looking_angle: side looking angle
        :param trajectory_angle: trajectory angle on GCS xy plane
        :param squint_angle: forward looking squint angle
        :return: lcs
        """
        # rotation around z(alpha)`and then new around x(beta)  and then around y(gamma)
        self.trajectory_angle = trajectory_angle  # store it to be used in the speed vector
        self.side_looking_angle = looking_angle  # looking angle from the down looking direction to the right of the azimuth path
        self.forward_squint_angle = squint_angle  # forward looking angle toward the walk path
        alpha = trajectory_angle
        beta = looking_angle
        gamma = -squint_angle
        self.x_s = np.array([
            cos(alpha) * cos(gamma) + sin(alpha) * sin(beta) * sin(gamma),
            sin(alpha) * cos(gamma) - cos(alpha) * sin(beta) * sin(gamma),
            -cos(beta) * sin(gamma)
        ])
        self.y_s = np.array([
            sin(alpha) * cos(beta),
            -cos(alpha) * cos(beta),
            sin(beta)
        ])
        self.z_s = np.array([
            -cos(alpha) * sin(gamma) + sin(alpha) * sin(beta) * cos(gamma),
            -sin(alpha) * sin(gamma) - cos(alpha) * sin(beta) * cos(gamma),
            -cos(beta) * cos(gamma)
        ])
        # local basis matrix (transformation matrix between local and canonical basis)
        self.Bs2c = np.column_stack((self.x_s, self.y_s, self.z_s))
        # and vice versa
        self.Bc2s = self.Bs2c.T

        # update velocity vector
        if hasattr(self, 'abs_v'):
            self.set_speed(self.abs_v)

        return self.x_s, self.y_s, self.z_s

    def set_speed(self, speed: float):
        """
        sets the velocity vector of the radar(assuming in first approximation that the velocity is constant
        :param speed:
        :param trajectory_angle:
        :return:
        """
        alpha = self.trajectory_angle
        self.abs_v = speed
        self.velocity = self.abs_v * np.array((np.cos(alpha), np.sin(alpha), 0))
        self.velocity.shape = (3, 1)  # column vector
        return self.velocity

    def set_initial_position(self, x: float, y: float, z: float):
        """
        initial position with respect to the global reference system
        :param x: x GCS coordinate
        :param y: y GCS coordinate
        :param z: altitude i.e. z GCS coordinate
        :return: array form of initial position
        """
        self.S_0 = np.array([[x], [y], [z]])  # column
        return self.S_0

    def get_lcs_of_point(self, point_target, t):
        """
        get the local coordinates of a target *!* in time *!* as seen by the radar antenna
        :param point_target: numpy array 3-d containing the target gc
        :param t: time axis
        :return: 3 by len(t) lcs array
        """
        target = point_target
        # transpose target vector
        target.shape = (3, 1)
        # gcs range vector
        r = np.ones_like(t) * target - self.get_position(t)
        # change to local
        return np.matmul(self.Bc2s, r)

    def get_range(self, point_target, t):
        """
        gets relative range between satellite and target in time
        :param point_target: numpy array of dimention 3 containing the gcs coordinates of the point target
        :param t:
        :return:
        """
        target = point_target
        target.shape = (3, 1)
        r = self.get_position(t) - target * np.ones_like(t)
        r = np.linalg.norm(r, axis=0)
        return r

    def orbital_speed(self):
        """
        ideal circular orbit speed approximation
        :return: the platform estimated speed
        """
        # the platform speed  # gravitational mu # earth radius
        radar_speed = np.sqrt(3.9860044189e14 / (6378e3 + self.S_0[2]))  # m/s
        return radar_speed


## FUNCITONS $$

## spherical to cartesian canonical functions ##


@jit(nopython=True)
def sph2cart(P: np.ndarray) -> np.ndarray:
    """ Convert a numpy array in the form [r,theta,phi] to a numpy array in the form [x,y,z]
  """
    r, theta, phi = 0, 1, 2
    x = P[r] * np.sin(P[theta]) * np.cos(P[phi])
    y = P[r] * np.sin(P[theta]) * np.sin(P[phi])
    z = P[r] * np.cos(P[theta])
    return np.array((x, y, z))


# rectangular to polar
@jit(nopython=True)
def cart2sph(P: np.ndarray) -> np.ndarray:
    """ Convert a numpy array in the form [x,y,z] to a numpy array in the form [r,theta,phi]
    """
    x, y, z = 0, 1, 2
    # r = np.linalg.norm(P)
    r = np.sqrt(P[x] ** 2 + P[y] ** 2 + P[z] ** 2).astype(np.float64)
    theta = np.where(r != 0, np.arccos(P[z] / r), 0).astype(np.float64)
    phi = np.where(r != 0, np.arctan2(P[y], P[x]), 0).astype(np.float64)
    return np.stack((r, theta, phi))


# polar to rectangular meshgrid
@jit(nopython=True, parallel=True)
def meshSph2cart(r_mesh, theta_mesh, phi_mesh):
    """

    :param r_mesh: r coordinates meshgrid
    :param theta_mesh: theta coordinates meshgrid
    :param phi_mesh: phi coordinates meshgrid
    :return: x, y, z in gcs
    """
    x = np.zeros_like(r_mesh).astype(np.float64)
    y = np.zeros_like(theta_mesh).astype(np.float64)
    z = np.zeros_like(phi_mesh).astype(np.float64)

    rows, columns = r_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            x = r_mesh[rr, cc] * np.sin(theta_mesh[rr, cc]) * np.cos(phi_mesh[rr, cc])
            y = r_mesh[rr, cc] * np.sin(theta_mesh[rr, cc]) * np.sin(phi_mesh[rr, cc])
            z = r_mesh[rr, cc] * np.cos(theta_mesh[rr, cc])

    return x, y, z


# Rectangular to spherical meshgrid
@jit(nopython=True, parallel=True)
def meshCart2sph(x_mesh, y_mesh, z_mesh):
    """
    from rec to sph
    :param x_mesh: x coordinates meshgrid
    :param y_mesh: y coordinates meshgrid
    :param z_mesh: z coordinates meshgrid
    :return: r_mesh, theta_mesh, phi_mesh
    """
    r_mesh = np.zeros_like(x_mesh).astype(np.float64)
    theta_mesh = np.zeros_like(x_mesh).astype(np.float64)
    phi_mesh = np.zeros_like(x_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            r_mesh[rr, cc] = np.sqrt(x_mesh[rr, cc] ** 2 + y_mesh[rr, cc] ** 2 + z_mesh[rr, cc] ** 2)
            theta_mesh[rr, cc] = np.where(r_mesh[rr, cc] != 0, np.arccos(z_mesh[rr, cc] / r_mesh[rr, cc]), 0)
            phi_mesh[rr, cc] = np.where(r_mesh[rr, cc] != 0, np.arctan2(y_mesh[rr, cc], x_mesh[rr, cc]), 0)

    return r_mesh, theta_mesh, phi_mesh


## cartesian changes of coordinates

# fast implementation for change of coordinates
@jit(nopython=True, parallel=True)
def mesh_lcs_to_gcs(x_mesh, y_mesh, z_mesh, Bs2c, S0):
    """

    :param x_mesh: lcs x coordinates meshgrid
    :param y_mesh: lcs y coordinates meshgrid
    :param z_mesh: lcs z coordinates meshgrid
    :param Bs2c: matrix of basis change from local to global
    :param S0: position of the local coordinate system
    :return: x, y, z in gcs
    """
    x = np.zeros_like(x_mesh).astype(np.float64)
    y = np.zeros_like(y_mesh).astype(np.float64)
    z = np.zeros_like(z_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            P = S0 + Bs2c @ np.array([[x_mesh[rr, cc]], [y_mesh[rr, cc]], [z_mesh[rr, cc]]])
            x[rr, cc], y[rr, cc], z[rr, cc] = P[0], P[1], P[2]

    return x, y, z


# fast implementation for change of coordinates
@jit(nopython=True, parallel=True)
def mesh_gcs_to_lcs(x_mesh, y_mesh, z_mesh, Bc2s, S0):
    """

    :param x_mesh: gcs x coordinates meshgrid
    :param y_mesh: gcs y coordinates meshgrid
    :param z_mesh: gcs z coordinates meshgrid
    :param Bc2s: matrix of basis change from global to local
    :param S0: position of the local coordinate system
    :return: x, y, z meshgrids in lcs
    """
    x = np.zeros_like(x_mesh).astype(np.float64)
    y = np.zeros_like(y_mesh).astype(np.float64)
    z = np.zeros_like(z_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            P = Bc2s @ (np.array([[x_mesh[rr, cc]], [y_mesh[rr, cc]], [z_mesh[rr, cc]]]) - S0)
            x[rr, cc] = P[0, 0]
            y[rr, cc] = P[1, 0]
            z[rr, cc] = P[2, 0]

    return x, y, z


## stationary phase points transformations

# find an azimuth distance given the incidence angle and the doppler shift
def mesh_doppler_to_azimuth(theta_mesh, doppler_mesh, lambda_c, v_s, h=500e3, re=6371e3, c=299792458.0):
    """
    from doppler shift to azimuth point, defined as the length of an arc on the circle with constant incidence angle on
    the spherical earth

    :param theta_mesh: incidence angle
    :param doppler_mesh: doppler shift
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :param c: optional, default speed of light
    :return: incidence angle - azimuth mesh
    """
    # closest approach range for every incidence angle value [Gebert]
    R0_mesh = re * (np.sqrt(cos(theta_mesh) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(theta_mesh))
    # cosine of earth-centric elevation angle of azimuth circle parallel to the orbital plane on the sphere
    cos_theta_e = (re + R0_mesh * cos(theta_mesh)) / (re + h)
    # collected equations
    A = (re + h) ** 2 + re ** 2
    B = 2 * (re + h) * re * cos_theta_e
    B[np.isnan(B)] = 0
    v = v_s / (re + h)
    # the absolute stationary phase time ( time - doppler relation ) is
    arg = ((lambda_c ** 2 * doppler_mesh ** 2 + np.sqrt(
        -4 * A * lambda_c ** 2 * v ** 2 * doppler_mesh ** 2 + 4 * B ** 2 * v ** 4 + lambda_c ** 4 * doppler_mesh ** 4)) / (
                   2 * B * v ** 2))
    # making sure it doesn't exceed 1
    # print(arg)
    arg[np.isnan(arg)] = 0
    arg = np.where(np.abs(arg) > 1, np.sign(arg), arg)
    # print(arg)
    t_k = np.abs(1 / v * np.arccos(arg))
    # sign retrieval
    t_k = np.where(doppler_mesh < 0, t_k, - t_k)
    # azimuth coordinate
    # print(cos_theta_e)
    Az_mesh = re * v * t_k * cos_theta_e
    # not considering the range walk. i.e az and incidence should change slightly with t_k
    return theta_mesh, Az_mesh, t_k


# returns the amplitude scaling of the stationary point in doppler
def stationary_phase_amplitude_multiplier(theta_mesh, stationary_time_mesh, lambda_c, v_s, h=500e3, re=6371e3,
                                          c=299792458.0):
    """
    given the incidence angle and stationary time mesh (associated to a doppler shift) finds the amplitude scaling for the spectrum
    :param theta_mesh: incidence angle
    :param stationary_time_mesh: stationary time points related to the doppler axis
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :param c: optional, default speed of light
    :return:
    """
    # closest approach range for every incidence angle value [Gebert]
    R0_mesh = re * (np.sqrt(cos(theta_mesh) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(theta_mesh))
    # cosine of earth-centric elevation angle of azimuth circle parallel to the orbital plane on the sphere
    cos_theta_e = (re + R0_mesh * cos(theta_mesh)) / (re + h)
    # collected equations
    A = (re + h) ** 2 + re ** 2
    B = 2 * (re + h) * re * cos_theta_e
    v = v_s / (re + h)
    # amplitude scaling (using the 2nd derivative of the range walk)
    sf_mesh = np.sqrt(lambda_c) * np.sqrt(np.abs((-2 * (A - B * cos(v * stationary_time_mesh)) ** (3 / 2)) /
                                                 (B * v ** 2 * (-2 * A * cos(v * stationary_time_mesh) + B * cos(
                                                     v * stationary_time_mesh) ** 2 + B))))
    return sf_mesh


def mesh_incidence_azimuth_to_gcs(incidence_mesh, azimuth_mesh, lambda_c, v_s, h=500e3, re=6371e3, c=299792458.0):
    """
    converts incidence angle - azimuth distance coordinates to xyz GCS coordinates
    :param incidence_mesh: incidence angle
    :param azimuth_mesh:  azimuth distance on the closeast approach incidence angle circle (spherical earth)
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param c: optional, default speed of light
    :return: x_mesh, y_mesh, z_mesh
    """
    # closest approach slant range
    R0_mesh = re * (np.sqrt(cos(incidence_mesh) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence_mesh))
    # cosine of earth-centric elevation angle of azimuth circle parallel to the orbital plane on the sphere
    cos_theta_e = (re + R0_mesh * cos(incidence_mesh)) / (re + h)
    # elevation coordinate of point
    theta_e = np.arccos(cos_theta_e) * np.sign(incidence_mesh)  # to consider also incidence angles behind nadir
    # azimuth angle coordinate
    theta_a = azimuth_mesh / (re * cos_theta_e)
    # x coordinate mesh
    x = re * cos_theta_e * sin(theta_a)
    # y coordinate mesh
    y = - re * sin(theta_e)
    # z coordinate mesh
    z = -re + re * cos_theta_e * cos(theta_a)
    return x, y, z


def mesh_incidence_time_to_incidence_azimuth(incidence_mesh, time_mesh, v_s, h=500e3, re=6371e3):
    """
    converts the slow time to azimuth ground displacement given the incidence angle
    :param incidence_mesh: incidence angle [rad]
    :param time_mesh: slow time
    :param v_s: satellite orbital speed
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :return: incidence, azimuth
    """
    # closest approach slant range
    R0_mesh = re * (np.sqrt(cos(incidence_mesh) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence_mesh))
    # cosine of earth-centric elevation angle of azimuth circle parallel to the orbital plane on the sphere
    cos_theta_e = (re + R0_mesh * cos(incidence_mesh)) / (re + h)
    # ground velocity
    v_g = re * cos_theta_e * v_s / (re + h)
    # azimuth ground distance covered
    azimuth_mesh = v_g * time_mesh
    return incidence_mesh, azimuth_mesh


# move this to design_function.
def nominal_doppler_bandwidth(antenna_length, incidence_angle, lambda_c, v_s, h=500e3, re=6371e3, c=299792458.0):
    """
    non squinted radar nominal doppler bandwidth (3-dB antenna Beamwidth) with some degree of approximation
    for spherical earth
    :param antenna_length: antenna length
    :param incidence_angle: broadside point incidence angle radians
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :param c: optional, default speed of light
    :return:
    """

    r0 = re * (np.sqrt(cos(incidence_angle) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence_angle))
    angle_az = np.arcsin(lambda_c / antenna_length)
    # we approximate the azimuth length illuminated by the antenna as
    d = r0 * np.tan(angle_az)

    # this corresponds to an integration time (at the incidence angle) of
    ## using the ground point elevation angle
    cos_theta_e = (re + r0 * cos(incidence_angle)) / (re + h)
    ## and the closest approach range point speed on ground
    vg = v_s / (re + h) * re * cos_theta_e
    it = d / vg

    # the doppler bandwidth is then
    A = (re + h) ** 2 + re ** 2
    B = 2 * (re + h) * re * cos_theta_e
    v = v_s / (re + h)
    doppler_band = 2 * (B * v * sin(v * it / 2) / (lambda_c * np.sqrt(A - B * cos(v * it / 2))))
    return doppler_band

def integration_time(antenna_length, incidence_angle, lambda_c, v_s, h=500e3, re=6371e3, c=299792458.0):
    """
    non squinted radar nominal integration time (3-dB antenna Beamwidth) with some degree of approximation
    for spherical earth
    :param antenna_length: antenna length
    :param incidence_angle: broadside point incidence angle radians
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :param c: optional, default speed of light
    :return: it
    """
    r0 = re * (np.sqrt(cos(incidence_angle) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence_angle))
    angle_az = np.arcsin(lambda_c / antenna_length)
    # we approximate the azimuth length illuminated by the antenna as
    d = r0 * np.tan(angle_az)

    # this corresponds to an integration time (at the incidence angle) of
    ## using the ground point elevation angle
    cos_theta_e = (re + r0 * cos(incidence_angle)) / (re + h)
    ## and the closest approach range point speed on ground
    vg = v_s / (re + h) * re * cos_theta_e
    it = d / vg
    return it


def incidence_angle_to_looking_angle(incidence_angle, h=500e3, re=6371e3):
    """

    :param incidence_angle: broadside incidence angle radians
    :param h: optional, satellite height from ground, default 500 km
    :param re: optional, spherical earth radius, default 6371 km
    :return:
    """
    r0 = re * (np.sqrt(cos(incidence_angle) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence_angle))
    # using the rule of cosines
    cos_theta_l = (r0 ** 2 + (re + h) ** 2 - re ** 2) / (2 * r0 * (re + h))
    return np.arccos(cos_theta_l)


def core_snr_spherical(radarGeo: RadarGeometry, uniap: UniformAperture, incidence, lambda_c, v_s, h=500e3, Bd_scaling=1,
                       re=6371e3, c=299792458.0):
    """

                        λ² ∙ G² · c · Bd · vg
    C = ----------------------------------------------------
        128 · π^3 · R0^4 · k · sin(η)· INT_Bd{ |H(fd)|² δfd}

    k, boltzmann constant
    R0 slant range
    η, incidence angle
    H(fd), doppler compression filter (depends on normalized gain pattern)
    G, peak antenna gain
    vg = vs / (Re + h) * Re * cos(θe), ground projected speed

    :param radarGeo: radar geometry object
    :param uniap: Antenna aperture object
    :param incidence: incidence angles vector
    :param lambda_c: wavelength
    :param v_s: satellite orbital speed (circular orbit)
    :param h: optional, satellite height from ground, default 500 km
    :param Bd_scaling: optional, default 1. increase or decrease the processed doppler bandwidth
    :param re: optional, spherical earth radius, default 6371 km
    :param c: optional, default speed of light
    :return:
    """
    # incidence angle from looking angle
    looking_angle = radarGeo.side_looking_angle
    r0 = cos(looking_angle) * (re + h) - np.sqrt(
        re ** 2 * cos(looking_angle) ** 2 + 2 * re * cos(looking_angle) ** 2 * h - 2 * re * h + cos(
            looking_angle) ** 2 * h ** 2 - h ** 2)
    cos_theta_i = - (-(re + h) ** 2 + re ** 2 + r0 ** 2) / (2 * r0 * re)
    theta_i = np.arccos(cos_theta_i)
    # print(looking_angle*180/np.pi)
    # print(theta_i*180/np.pi)

    # 1 Doppler Bandwidth
    Bd = nominal_doppler_bandwidth(uniap.L, theta_i, lambda_c, v_s, h) * Bd_scaling
    # print(Bd)
    # print(v_s)
    # print(lambda_c)
    # 2 doppler axis
    doppler = np.linspace(-Bd / 2, Bd / 2, 513)

    # 3 meshgrids
    I, D = np.meshgrid(incidence, doppler)
    # incidence azimuth stationary time
    I, A, Tk = mesh_doppler_to_azimuth(I, D, lambda_c, v_s, h)
    # GCS
    X, Y, Z = mesh_incidence_azimuth_to_gcs(I, A, lambda_c, v_s, h)
    R0_mesh = re * (np.sqrt(cos(I) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(I))
    # cosine of earth-centric elevation angle of azimuth circle parallel to the orbital plane on the sphere
    cos_theta_e = (re + R0_mesh * cos(I)) / (re + h)
    # LCS
    Xl, Yl, Zl = mesh_gcs_to_lcs(X, Y, Z, radarGeo.Bc2s, radarGeo.S_0)
    # LCS spherical
    R, T, P = meshCart2sph(Xl, Yl, Zl)

    # 4 antenna pattern
    Gain = uniap.mesh_gain_pattern_theor(T, P) / uniap.max_gain()
    Gain = np.where(np.isnan(Gain), 1, Gain)

    # 5 Integrand
    # the matched filter amplitude
    H = 1 / (stationary_phase_amplitude_multiplier(I, Tk, lambda_c, v_s, h) * Gain)
    w_range = integrate.simps(H ** 2, D, axis=0)
    # print(w_range)
    # 6 Core SNR equation
    # Boltzman constant
    k_boltz = 1.380649E-23  # J/K
    # the sin of the incidence angle at each ground range point
    sin_theta_i = sin(incidence)
    # The range at each ground range point
    r0 = re * (np.sqrt(cos(incidence) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(incidence))
    max_gain = uniap.max_gain()
    # print(max_gain)
    vg = v_s / (re + h) * re * cos_theta_e[0, :]
    # the equation is then: (equivalent to the above, just simplified)
    SNR_core = lambda_c ** 2 * max_gain ** 2 * c * Bd * vg / (
            128 * np.pi ** 3 * r0 ** 4 * k_boltz * sin_theta_i * w_range)

    # azimuth resolutions
    daz = vg / Bd

    return SNR_core, daz


def snr_error(beta, theta_ne, theta_fe, radarGeo: RadarGeometry, uniAp: UniformAperture):
    """
    error proportional to the asymmetry between snr at ne and at fe given points

    :param beta: radar Looking Angle [radians]
    :param theta_ne: incidence angle Near End [radians]
    :param theta_fe: incidence angle Far End [radians]
    :param radarGeo:  radarGeometry object
    :param uniAp:  uniform aperture object
    :return: error to be minimized
    """
    incidence = np.array([theta_ne, theta_fe])
    # 2 radar object
    radarGeo.set_rotation(beta, 0, 0)
    # 3 parameters
    wavel = uniAp.c / uniAp.freq
    v_s = radarGeo.orbital_speed()
    h = radarGeo.S_0[2]
    # 4 SNR core
    SNR_core, daz = core_snr_spherical(radarGeo, uniAp, incidence, wavel, v_s, h)
    # 5 Error
    e = SNR_core[1] - SNR_core[0]
    return e


def pd_error(beta, theta_ne, theta_fe, radarGeo: RadarGeometry, uniAp: UniformAperture, C_multiplier, B, pfa, aship,
             mean, var, c=299792458.0):
    """
    error proportional to the asymmetry between snr at ne and at fe given points

    :param beta: radar Looking Angle [radians]
    :param theta_ne: incidence angle Near End [radians]
    :param theta_fe: incidence angle Far End [radians]
    :param radarGeo:  radarGeometry object
    :param uniAp:  uniform aperture object
    :param C_multiplier: Nesz = C_multiplier / coreSNR
    :param B: noise bandwidth
    :param pfa: Probability of false alarm
    :param aship: ship area
    :param c: optional, default speed of light
    :return: error to be minimized
    """
    # 1 start finding the nesz
    incidence = np.array([theta_ne, theta_fe])
    # 2 radar object
    radarGeo.set_rotation(beta, 0, 0)
    # 3 parameters
    wavel = uniAp.c / uniAp.freq
    v_s = radarGeo.orbital_speed()
    h = radarGeo.S_0[2]
    # 4 SNR core
    SNR_core, daz = core_snr_spherical(radarGeo, uniAp, incidence, wavel, v_s, h)
    # 5 NESZ
    nesz = C_multiplier / SNR_core
    # 6 pd
    acell = daz * c / (2 * sin(incidence) * B)
    Pd = pd_from_nesz_res(nesz, acell, pfa, aship, mean, var)
    # 5 Error
    e = Pd[0] - Pd[1]
    return e
