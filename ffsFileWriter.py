# Simone Mencarelli
# September 2023
# this file contains a function to pack a far field pattern into a CST ffs file
# the geometry is fixed, single frequency only
import numpy as np


# %% User input
#
# filename = 'out.ffs'
# theta = np.random.rand(20)
# phi = np.random.rand(20)
# e_theta = np.random.rand(20) + 1j * np.random.rand(20)
# e_phi = np.random.rand(20) + 1j * np.random.rand(20)
# radiated_power = 1
# accepted_power = 2
# stimulated_power = 3
# frequency = 4

# %% function
def ffsWrite(theta, phi, e_theta, e_phi,
             num_phi, num_theta,
             filename='out.ffs',
             radiated_power=1,
             accepted_power=1,
             stimulated_power=1,
             frequency=10e9):
    """
    export a pattern to ffs format
    :param theta: least significant , when unraveling the meshgrid this has to vary faster than theta
    :param phi: most significant , when unraveling the meshgrid this has to vary slower than theta
    :param e_theta: complex 1d
    :param e_phi: complex 1d
    :param num_phi:
    :param num_theta:
    :param filename: something.ffs
    :param radiated_power:
    :param accepted_power:
    :param stimulated_power:
    :param frequency:
    :return:
    """
    # %% Writer
    # preamble string
    pre = '// CST Farfield Source File\n \n// Version:\n3.0 \n\n// Data Type\nFarfield \n\n// #Frequencies\n1 \n\n// Position\n0.000000e+00 0.000000e+00 0.000000e+00 \n\n// zAxis\n0.000000e+00 0.000000e+00 1.000000e+00 \n\n// xAxis\n1.000000e+00 0.000000e+00 0.000000e+00 \n'
    # other preambles
    power = '\n// Radiated/Accepted/Stimulated Power , Frequency \n'
    samples = '\n\n// >> Total #phi samples, total #theta samples \n'
    pattern = '\n// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi): \n'

    with open(filename, 'w') as file:
        file.write(pre)
        file.write(power)
        file.write("{:.6e}".format(radiated_power) + '\n')
        file.write("{:.6e}".format(accepted_power) + '\n')
        file.write("{:.6e}".format(stimulated_power) + '\n')
        file.write("{:.6e}".format(frequency) + '\n')
        file.write(samples)
        file.write(str(num_phi) + ' ' + str(num_theta) + '\n')
        file.write(pattern)
        for i in range(len(theta)):
            file.write("{0:10.4f} {1:10.4f} {2:16.8e} {3:16.8e} {4:16.8e} {5:16.8e} \n".format(phi[i],
                                                                                               theta[i],
                                                                                               np.real(e_theta[i]),
                                                                                               np.imag(e_theta[i]),
                                                                                               np.real(e_phi[i]),
                                                                                               np.imag(e_phi[i])))
