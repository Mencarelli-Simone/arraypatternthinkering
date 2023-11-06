#####################################################
# Useful functions for SAR system initial design
# by Simone Mencarelli
# August 2022
#####################################################
import numpy
from numpy import arctan, tan, sin, cos, vectorize, arctan2, arccos
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf


# RANGE FROM INCIDENCE ANGLE #
# an explanation of how this was derived can be found in the
# N Gerbert phd thesis
def range_from_theta(Theta, h=500e3, re=6371e3):
    """
    returns the slant range and spherical earth ground range given the incidence angle
    :param Theta: incidence angle in degrees, must be between 0 and 90
    :param h: orbital height, default 500 km
    :param re: earth radius, default 6,371 km
    :return: (slant_range, ground_range)
    """
    # # step 1 find the slant range
    theta = np.pi * Theta / 180  # better in radians
    r = re * (np.sqrt(cos(theta) ** 2 + 2 * h / re + h ** 2 / re ** 2) - cos(theta))
    rg = arccos((re + r * cos(theta)) / (re + h)) * re
    # # finally return all
    return r, rg


# ground range from slant range:
def range_slant_to_ground(rs, h=500e3, re=6371e3):
    """
    returns the spherical earth ground range and incidence angle associated to a slant range
    :param rs: slant range of a point
    :param h: orbital height, default 500 km
    :param re: earth radius, default 6,371 km
    :return: (ground_range, theta)
    """
    # using Carnot's theorem:
    beta = arccos(((re + h) ** 2 - re ** 2 + rs ** 2) / (2 * (re + h) * rs))
    alpha = arccos(((re + h) ** 2 + re ** 2 - rs ** 2) / (2 * (re + h) * re))
    # the incidence angle is the sum of the looking angle and the
    # angle subtended by the ground range by earth's circumference
    theta = alpha + beta
    rg = alpha * re
    return rg, theta


# ground range from slant range:
def range_ground_to_slant(rg, h=500e3, re=6371e3):
    """
    returns the spherical earth ground range and incidence angle associated to a slant range
    :param rg: ground range of a point
    :param h: orbital height, default 500 km
    :param re: earth radius, default 6,371 km
    :return: slant_range
    """
    alpha = rg / re
    # using Pitagora's
    rs = np.sqrt((re * sin(alpha)) ** 2 + (h + re - re * cos(alpha)) ** 2)
    return rs


# pulse repetition intervall for largest possible swath
# a step by step derivation can be found in the notebook
# 'PRIcalculator.ipynb'
def pri_max_swath(theta, wa, freq=10e9, c=299792458, nefe=False, slant=False):
    """
    compute the prf to achieve the widest possible swath
    doesn't account for nadir return cancellation
    :param theta: scene center incidence angle degrees
    :param wa: antenna width in elevation
    :param freq: carrier frequency, default: 10 GHz
    :param c: propagation speed, default: light
    :return: PRI, ground_swath
    """
    # wavelength
    wavel = c / freq

    # # step 1 elevation limits
    theta1 = (theta * np.pi / 180) + wavel / (2 * wa)
    theta2 = (theta * np.pi / 180) - wavel / (2 * wa)

    # # step 2 slant range limits
    r1s, r1g = range_from_theta(theta1 * 180 / np.pi)
    r2s, r2g = range_from_theta(theta2 * 180 / np.pi)

    # slant swath
    delta_r_s = r1s - r2s

    # # step 3 intial PRI
    PRI = float(2 * delta_r_s / c)

    # # step 4 impulse order determination
    # average slant range
    rs = np.average((r1s, r2s))
    order = np.floor(2 * rs / (PRI * c))

    # # step 5 adjusted PRI
    PRI1 = 2 * rs / (c * (.5 + order))

    # # step 6 adjusted near end and far end range and swath
    r2s1 = order * c * PRI1 / 2
    r1s1 = (order + 1) * c * PRI1 / 2

    # # step 7 adjusted ground ranges
    r1g1, theta1 = range_slant_to_ground(r1s1)
    r2g1, theta2 = range_slant_to_ground(r2s1)
    rg1 = r1g1 - r2g1

    if nefe:
        if slant:
            return PRI1, np.array([r2s1, r1s1])
        else:
            return PRI1, np.array([r2g1, r1g1])
    else:
        return PRI1, rg1


# closest valid point in the timing diagram for nadir return suppression
def closest_nadir_null(pri, rne, h=500e3, c=299792458, end='near'):
    """
    corrects theta, the swath and pri to the closest valid point
    :param pri: current Pulse Repetition Intervall
    :param rne: current SLANT range Near end or far end value
    :param h: optional, satellite height, default 500km
    :param c: optional, default speed of light
    :return: theta', pri', ground_swath= [R_ne',R_fa']
    """
    # 1 find closest valid PRI
    m = int(np.round(2 * h / (c * pri)))
    pri1 = h * 2 / (m * c)
    # 2 find closest valid near end range at that PRI
    n = int(np.round(2 * rne / (pri1 * c)))
    if end == 'far':
        n = int(np.round(2 * rne / (pri1 * c))) - 1
    slant_swath_2 = np.array([n * pri1 * c / 2, (n + 1) * pri1 * c / 2])
    # 3 convert to theta and ground range
    ground_swath_2, theta_1 = range_slant_to_ground(slant_swath_2, h)
    # Just need the broadside and in degs
    theta_1 = (theta_1[0] + theta_1[-1]) * 180 / (2 * np.pi)
    return theta_1, pri1, ground_swath_2


def pd_from_nesz_res(nesz, acell, pfa, aship, mean, var):
    """
    compute the probability of detection given, resolution area, probability of false alarm, ship area and log normal distribution properties
    :param nesz: Noise equivalent sigma zero
    :param acell: resolution area on ground
    :param pfa: Probability of false alarm
    :param aship: ship area
    :return: pd
    """
    Thresh = - nesz * np.log(pfa * acell / aship)
    # Probability of detection
    Pd = 1 - (1 / 2 + 1 / 2 * erf((np.log(Thresh) - mean) / (np.sqrt(2 * var)))) ** (aship / acell)
    return Pd

def find_bandwidth(La, Theta, Ares, c=299792458):
    """
    design equation for bandwidth selection
    :param La: Antenna length
    :param Theta: incidence angle in degrees
    :param Ares: Resolution Area
    :return: Bandwidth
    """
    return c * La / (4 * np.sin(Theta * np.pi / 180) * Ares)