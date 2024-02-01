# Simone Mencarelli
# September 2023
# This file contains a class with the same interface of the Aperture class in radartools.farField
# The pattern however is loaded from a CST ffs file and provided for any theta phi coordinate
# by means of an interpolator, namely the sphere_interp in interpolator_v2.py

# %% includes
import numpy as np
from numpy import sin, cos, pi, tan, arcsin, arccos, arctan, sqrt, exp, log10
from interpolator_v2 import sphere_interp


# %% functions

# far field source loader (ffsFileReader script)
def ffsLoader(filename):
    # %% variables to read from file
    # header
    frequencies = 0
    position = [0, 0, 0]
    radiatedPower = 0
    acceptedPower = 0
    stimulatedPower = 0
    frequency = 0
    phiSamples = 0
    thetaSamples = 0

    # data matrix
    dataMatrix = None  # size is going to be initialized after reading header

    # %% Read file in ram
    with open(filename, 'r') as file:
        content = file.read()
    content = content.split('\n')

    # %% parse
    # parse just header
    for i in range(len(content)):
        if "Frequencies" in content[i]:
            frequencies = int(content[i + 1])
            i += 1
        if "Position" in content[i]:
            pos = content[i + 1]
            pos = pos.split(' ')
            position = np.array(pos[0:3], dtype=float).reshape((3, 1))
            i += 1
        if "Radiated/Accepted/Stimulated Power , Frequency" in content[i]:
            radiatedPower = float(content[i + 1])
            acceptedPower = float(content[i + 2])
            stimulatedPower = float(content[i + 3])
            frequency = float(content[i + 4])
        if "Total #phi samples, total #theta samples" in content[i]:
            sam = content[i + 1]
            sam = sam.split(' ')
            phiSamples = int(sam[0])
            thetaSamples = int(sam[1])
            break

    # %% data parsing
    # Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi)
    dataMatrix = np.zeros((phiSamples * thetaSamples, 6))

    for i in range(len(content)):
        if "Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi)" in content[i]:
            for j in range(phiSamples * thetaSamples):
                dataline = content[i + j + 1]
                dataline = dataline.split(" ")
                if len(dataline) == 0:  # eof
                    break
                while "" in dataline:
                    dataline.remove("")
                dataline = np.array(dataline[0:6], dtype=float).reshape((1, 6))
                dataMatrix[j, :] = dataline

    # %% turn the data into meshgrids
    # theta component
    E_Theta = dataMatrix[:, 2] + 1j * dataMatrix[:, 3]
    E_Theta = E_Theta.reshape((phiSamples, thetaSamples))
    # phi component
    E_Phi = dataMatrix[:, 4] + 1j * dataMatrix[:, 5]
    E_Phi = E_Phi.reshape((phiSamples, thetaSamples))
    # coordinates
    Phi = dataMatrix[:, 0]
    Phi = Phi.reshape((phiSamples, thetaSamples))
    Theta = dataMatrix[:, 1]
    Theta = Theta.reshape((phiSamples, thetaSamples))

    return Phi, Theta, E_Phi, E_Theta, phiSamples, thetaSamples, radiatedPower, stimulatedPower, acceptedPower


def ffeLoader(filename):
    # %% variables to read from file
    # filename = 'EyPattern_NoDistortion.ffe'
    # header
    frequencies = 0
    position = [0, 0, 0]
    radiatedPower = 1
    acceptedPower = 1
    stimulatedPower = 1
    frequency = 0
    phiSamples = 1
    thetaSamples = 1

    # data matrix
    dataMatrix = None  # size is going to be initialized after reading header

    # %% Read file in ram
    with open(filename, 'r') as file:
        content = file.read()
    content = content.split('\n')

    # %% parse
    # parse just header
    for i in range(len(content)):
        if "Frequencies" in content[i]:
            frequencies = int(content[i + 1])
            i += 1
        if "Position" in content[i]:
            pos = content[i + 1]
            pos = pos.split(' ')
            position = np.array(pos[0:3], dtype=float).reshape((3, 1))
            i += 1
        if "Radiated/Accepted/Stimulated Power , Frequency" in content[i]:
            radiatedPower = float(content[i + 1])
            acceptedPower = float(content[i + 2])
            stimulatedPower = float(content[i + 3])
            frequency = float(content[i + 4])
        if "#Frequency: " in content[i]:
            id = content[i].find(": ")
            frequency = float(content[i][id + 2:])
        if "#No. of Theta Samples: " in content[i]:
            id = content[i].find(": ")
            thetaSamples = int(content[i][id + 2:])
        if "#No. of Phi Samples: " in content[i]:
            id = content[i].find(": ")
            phiSamples = int(content[i][id + 2:])
            break

    # %% data parsing
    # Theta, Phi, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi)
    dataMatrix = np.zeros((phiSamples * thetaSamples, 9))

    for i in range(len(content)):
        if "Re(Etheta)" in content[i]:
            for j in range(phiSamples * thetaSamples):
                dataline = content[i + j + 1]
                dataline = dataline.split(" ")
                if len(dataline) < 9:  # eof
                    print('eof')
                    break
                else:
                    while "" in dataline:
                        dataline.remove("")
                    dataline = np.array(dataline[0:9], dtype=float).reshape((1, 9))
                    dataMatrix[j, :] = dataline.astype('float')

    # %% turn the data into meshgrids
    # theta component
    E_Theta = dataMatrix[:, 2] + 1j * dataMatrix[:, 3]
    E_Theta = E_Theta.reshape((phiSamples, thetaSamples))
    # phi component
    E_Phi = dataMatrix[:, 4] + 1j * dataMatrix[:, 5]
    E_Phi = E_Phi.reshape((phiSamples, thetaSamples))
    # coordinates
    Phi = dataMatrix[:, 1]
    Phi = Phi.reshape((phiSamples, thetaSamples))
    Theta = dataMatrix[:, 0]
    Theta = Theta.reshape((phiSamples, thetaSamples))
    ## todo comment when fixed rotation in phi of 90deg
    # E_Theta = np.roll(E_Theta,int(phiSamples/4),axis=0)
    # E_Phi = np.roll(E_Phi, int(phiSamples / 4), axis=0)
    # E_Theta = np.flip(E_Theta,0)
    E_Phi = np.flip(E_Phi, 0)
    D_tot = dataMatrix[:, 8]
    D_tot = D_tot.reshape((phiSamples, thetaSamples))
    # radiated power
    G = 2 * np.pi * (np.abs(E_Theta) ** 2 + np.abs(E_Phi) ** 2) / (120 * np.pi * radiatedPower)
    norm = np.max(D_tot) / np.max(G)
    radiatedPower = 1. / norm
    return Phi, Theta, E_Phi, E_Theta, phiSamples, thetaSamples, radiatedPower, stimulatedPower, acceptedPower


# Aperture class for interfacing cst pattern
class Aperture:
    def __init__(self, filename):
        """
        initialization method, it requires a far field file
        :param filename: CST ffs file
        :return:
        """
        if filename[-4:].lower() == '.ffs':
            # load far field
            (Phi, Theta, E_Phi, E_Theta, phiSamples, thetaSamples,
             radiatedPower, stimulatedPower, acceptedPower) = ffsLoader(filename)
        elif filename[-4:].lower() == '.ffe':
            print('loading ffe pattern')
            # load far field
            (Phi, Theta, E_Phi, E_Theta, phiSamples, thetaSamples,
             radiatedPower, stimulatedPower, acceptedPower) = ffeLoader(filename)
        else:
            print('Error: file extension unknown')

        # compute directive gain
        self.G = np.array(
            2 * np.pi * (np.abs(E_Theta) ** 2 + np.abs(E_Phi) ** 2) / (120 * np.pi * radiatedPower),
            dtype=float)
        self.E_theta = E_Theta
        self.E_phi = E_Phi
        self.radiatedPower = radiatedPower
        # store relevant parameters
        self.Theta = Theta * np.pi / 180  # I use radians
        self.Phi = Phi * np.pi / 180
        self.phiSamples = phiSamples
        self.thetaSamples = thetaSamples
        # 3 compile the interpolator function
        theta = np.linspace(0, np.pi / 2, 5)
        phi = np.linspace(0, 2 * np.pi, 6)
        T, P = np.meshgrid(theta, phi)
        print('compiling')
        ginterp = self.mesh_gain_pattern(T, P, cubic=True)
        print('done')

    def mesh_gain_pattern(self, theta_mesh: np.ndarray, phi_mesh: np.ndarray, interpolant=None, cubic=True):
        """
        retruns the gain pattern at the specified meshgrid points in spherical coordinates.
        :param theta_mesh: Theta coordinates
        :param phi_mesh: Phi coordinates
        :param cubic: default True: bicubic interpolation utilised, False: linear interpolation.
        :return:
        """
        if interpolant is None:
            interpolant = self.G.T
        # phi_mesh = phi_mesh % (2 * np.pi)
        # spherical coordinates rearranged for negative theta
        phi_mesh = np.where(theta_mesh < 0, (phi_mesh + np.pi) % (np.pi * 2), phi_mesh)
        theta_mesh = np.abs(theta_mesh)
        # need to create an outpattern for some reason
        outpattern = np.zeros_like(theta_mesh).reshape(-1).astype('float')
        # theta and phi axes (origins of the meshgrid, non-uniform sampling not allowed)
        theta = self.Theta[0, :].reshape(-1)
        phi = self.Phi[:, 0].reshape(-1)
        # The vectors need to be flattened hence reshape(-1) except pattern
        outpattern = sphere_interp(theta_mesh.reshape(-1), phi_mesh.reshape(-1),
                                   theta, phi,
                                   interpolant, outpattern, cubic)
        outpattern = np.where(outpattern < 0, 0, outpattern)
        # reshape to desired format
        return outpattern.reshape(np.shape(theta_mesh))

    def co_cross_pol(self, polarization='x', E_theta=None, E_phi=None):
        """
        extracts the co and cross polarized components of the far field using Ludwig3 definition
        :param theta: theta angle
        :param phi: phi angle same shape of theta
        :param polarization: polarization of the co polarized component x or y (default x)
        :return:
        """
        shape = self.Theta.shape
        theta = self.Theta.reshape(-1)
        phi = self.Phi.reshape(-1)
        # compute the far field
        if E_theta is None or E_phi is None:
            E_theta, E_phi = self.E_theta, self.E_phi
        E_theta = E_theta.reshape(-1)
        E_phi = E_phi.reshape(-1)
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

    def co_cross_polar_gain(self, theta, phi, polarization='x', cubic=True):
        """
        returns copolar and crosspolar gain at the specified theta and phi coordinates
        :param theta:
        :param phi:
        :param polarization: x or y, copolar polarization (Ludwig III)
        :param cubic: interpolation setting cubic (True, default) or linear (False)
        :return:
        """
        # E fileds co cross
        E_co, E_cross = self.co_cross_pol(polarization, self.E_theta, self.E_phi)
        # gain co cross
        G_co = 2 * np.pi * (np.abs(E_co) ** 2) / (120 * np.pi * self.radiatedPower)
        G_cross = 2 * np.pi * (np.abs(E_cross) ** 2) / (120 * np.pi * self.radiatedPower)
        G_cross = G_cross.reshape(np.shape(self.G))
        G_co = G_co.reshape(np.shape(self.G))
        # interpolate
        G_co = self.mesh_gain_pattern(theta, phi, interpolant=G_co.T, cubic=cubic)
        G_cross = self.mesh_gain_pattern(theta, phi, interpolant=G_cross.T, cubic=cubic)
        return G_co, G_cross

    def max_gain(self):
        """
        :return: the peak (broadside) gain of pattern
        """
        max_g = np.max(self.G)
        return max_g


# %% testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1 create an aperture with the farfield file
    antenna = Aperture('lyceanem/NormalAntenna.ffe')
    #  Pass

    # 2 plot the original datum
    fig, ax = plt.subplots(1)
    ax.pcolormesh(antenna.Theta, antenna.Phi, 10 * np.log10(antenna.G), vmin=-50, vmax=50)
    plt.show()
    # Pass

    # %% 3 create a second meshgrid
    theta = np.linspace(0, np.pi / 2, 3501)
    phi = np.linspace(0, 2 * np.pi, 3500)
    T, P = np.meshgrid(theta, phi)
    ginterp = antenna.mesh_gain_pattern(T, P, cubic=True)
    ginterp = np.where(ginterp < 0, 0, ginterp)
    plt.show()
    # Pass

    # %%4 plot the output
    fig, ax = plt.subplots(1)
    ax.pcolormesh(T, P, 10 * np.log10(ginterp), vmin=-50, vmax=50)
    plt.show()
    # Pass

    # %% 6 same mesh and plot difference
    theta = np.linspace(0, np.pi / 2, antenna.thetaSamples)
    phi = np.linspace(0, 2 * np.pi, antenna.phiSamples)
    T, P = np.meshgrid(theta, phi)
    ginterp = antenna.mesh_gain_pattern(T, P, cubic=True)
    diff = ginterp - antenna.G
    fig, ax = plt.subplots(1)
    ax.pcolormesh(T, P, 10 * np.log10(diff))
    plt.show()
    print(np.max(np.abs(diff)))
    print(np.sum(np.abs(diff)))
    # # Pass
