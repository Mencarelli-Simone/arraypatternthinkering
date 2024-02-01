# Simone Mencarelli
# updated on September 2023
### !!!! Don't touch anything it works !!!!


import matplotlib.pyplot as plt
import numpy as np
from numba import prange, jit


@jit(nopython=True, parallel=True)  # faster without jit
def sphere_interp(theta_out, phi_out, theta_ax, phi_ax, pattern, out_pattern, cubic: bool = True):
    """

    :param theta_out: array_like 1d theta output points.  between 0 and pi
    :param phi_out: array_like 1d phi output points
    :param theta_ax: ordered homogeneously sampled theta axis. between 0 and pi
    :param phi_ax: ordered homogeneously sampled theta axis
    :param pattern: data to interpolate 2-d matrix
    :param cubic: optional, set to false to use rectangular
    :return: interpolated pattern array_like 1d as theta and phi
    """
    # output array
    # out_pattern = np.zeros_like(theta_out).astype('float')
    # explicit casting
    # pattern = pattern.astype('float')
    # print('compiled')

    flag = 0

    # find the min max step of axes
    theta_min = theta_ax[0]
    theta_max = theta_ax[-1]
    theta_step = abs((theta_ax[-1] - theta_ax[0]) / (len(theta_ax) - 1))

    # unwrapping phi
    phi_ax = np.where(phi_ax < 0, np.pi * 2 + phi_ax, phi_ax)
    phi_min = phi_ax[0]
    phi_max = phi_ax[-1]
    phi_step = (phi_ax[-1] - phi_ax[0]) / (len(phi_ax) - 1)

    # check the out samples are within the samples of the original pattern (nope, clashes with round 1 case)
    # if np.max(phi_out) > phi_max or np.max(theta_out) > theta_max or np.min(phi_out) < phi_min or np.min(
    #         theta_out) < phi_min:
    #     print("Error desired point out of boundaries")
    #     print(np.max(phi_out), np.max(theta_out), np.min(phi_out), np.min(theta_out))
    #     print(phi_max, theta_max, phi_min, phi_min)
    #     return

    phi_out = np.where(phi_out < 0, np.pi * 2 + phi_out, phi_out)
    # find 0 1 2 3 indices
    # print('finding indices')
    theta_idx_1 = np.floor((theta_out % (2 * np.pi) - theta_min) / theta_step).astype(np.int64)
    theta_idx_0 = theta_idx_1 - 1
    theta_idx_2 = theta_idx_1 + 1
    theta_idx_3 = theta_idx_1 + 2

    phi_idx_1 = np.floor((phi_out % (2 * np.pi) - phi_min) / phi_step).astype(np.int64)
    phi_idx_0 = phi_idx_1 - 1
    phi_idx_2 = phi_idx_1 + 1
    phi_idx_3 = phi_idx_1 + 2

    # print('checking indices')
    # check edges and return eventual errors
    maxidx = max(theta_idx_0.max(), theta_idx_1.max(), theta_idx_2.max(), theta_idx_3.max())
    minidx = min(theta_idx_0.min(), theta_idx_1.min(), theta_idx_2.min(), theta_idx_3.min())
    if maxidx >= len(theta_ax) + 2 or minidx < 0:
        # circular behaviour in phi
        # full phi axis last sample is the first (odd samples)
        if np.round((phi_max - phi_min) % (2 * np.pi)) == 0 and len(phi_ax) % 2 == 1:
            print('case 1')  # this is wrong, changing phi is not sufficient, we need to act on the interpolator
            # set a flag to tell the interpolator what to do
            flag = 1
            # # first sample
            # phi_idx_0 = np.where(theta_idx_0 < 0, (phi_idx_0 + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1),
            #                      phi_idx_0).astype(np.int64)
            # theta_idx_0 = np.abs(theta_idx_0).astype(np.int64)
            # # last samples
            # phi_idx_2 = np.where(theta_idx_2 > len(theta_ax) - 1,
            #                      (phi_idx_2 + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1), phi_idx_2).astype(np.int64)
            # phi_idx_3 = np.where(theta_idx_3 > len(theta_ax) - 1,
            #                      (phi_idx_3 + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1), phi_idx_3).astype(np.int64)
            # fold back in theta and make negative for flipping of phi afterwards
            theta_idx_2 = np.where(theta_idx_2 > len(theta_ax) - 1, -(2 * len(theta_ax) - 2 - theta_idx_2),
                                   theta_idx_2).astype(np.int64)
            theta_idx_3 = np.where(theta_idx_3 > len(theta_ax) - 1, -(2 * len(theta_ax) - 2 - theta_idx_3),
                                   theta_idx_3).astype(np.int64)

        # full phi axis last sample is one step before a full circle (even samples)
        elif np.round((2 * np.pi - (phi_max - phi_min) % (2 * np.pi)) / phi_step) == 1 and len(phi_ax) % 2 == 0:
            print('case 2')
            flag = 2
            ## first sample
            # phi_idx_0 = np.where(theta_idx_0 < 0, (phi_idx_0 + len(phi_ax) / 2) % len(phi_ax), phi_idx_0).astype(
            #     np.int64)
            # theta_idx_0 = np.abs(theta_idx_0).astype(np.int64)
            # # last samples
            # phi_idx_2 = np.where(theta_idx_2 > len(theta_ax), (phi_idx_2 + (len(phi_ax)) / 2) % len(phi_ax),
            #                      phi_idx_2).astype(np.int64)
            # phi_idx_3 = np.where(theta_idx_3 > len(theta_ax), (phi_idx_3 + (len(phi_ax)) / 2) % len(phi_ax),
            #                      phi_idx_3).astype(np.int64)
            # fold back in theta and make negative for phi flipping afterwards
            theta_idx_2 = np.where(theta_idx_2 > len(theta_ax) - 1, - (2 * len(theta_ax) - 2 - theta_idx_2),
                                   theta_idx_2).astype(np.int64)
            theta_idx_3 = np.where(theta_idx_3 > len(theta_ax) - 1, - (2 * len(theta_ax) - 2 - theta_idx_3),
                                   theta_idx_3).astype(np.int64)

        # incomplete phi axis, can't implement circular behaviour
        else:
            print('case 3')
            # retain index for edge values
            theta_idx_0 = np.where(theta_idx_0 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_0)
            theta_idx_1 = np.where(theta_idx_1 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_1)
            theta_idx_2 = np.where(theta_idx_2 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_2)
            theta_idx_3 = np.where(theta_idx_3 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_3)

            theta_idx_0 = np.where(theta_idx_0 < 0, 0, theta_idx_0)
            theta_idx_1 = np.where(theta_idx_1 < 0, 0, theta_idx_1)
            theta_idx_2 = np.where(theta_idx_2 < 0, 0, theta_idx_2)
            theta_idx_3 = np.where(theta_idx_3 < 0, 0, theta_idx_3)

    maxidx = max(phi_idx_0.max(), phi_idx_1.max(), phi_idx_2.max(), phi_idx_3.max())
    minidx = min(phi_idx_0.min(), phi_idx_1.min(), phi_idx_2.max(), phi_idx_3.min())
    # circular behavior in phi
    if maxidx >= len(phi_ax) or minidx < 0:
        # print(" Sphere interpolation error, phi index outside boundaries")
        # print(maxidx, minidx, phi_min, phi_max, len(phi_ax), phi_step)
        if np.round(np.abs((phi_max - phi_min) - 2 * np.pi)) == 0:  # last sample is the first sample
            print('case-1')
            phi_idx_0 = np.where(phi_idx_0 < 0, (len(phi_ax) + phi_idx_0 - 1) % len(phi_ax - 1),
                                 phi_idx_0)  # can't use negative indexes because of the special case
            phi_idx_2 = np.where(phi_idx_3 > len(phi_ax) - 1, phi_idx_2 % len(phi_ax - 1), phi_idx_2)
            phi_idx_3 = np.where(phi_idx_3 > len(phi_ax) - 1, phi_idx_3 % len(phi_ax - 1), phi_idx_3)
        # last sample is just before the first sample but separated by a step
        elif np.round(np.abs((phi_max - phi_min) - 2 * np.pi) / phi_step) == 1:
            print('case-2')
            phi_idx_0 = np.where(phi_idx_0 < 0, (len(phi_ax) + phi_idx_0) % len(phi_ax),
                                 phi_idx_0)  # can't use negative indexes because of the special case
            phi_idx_2 = np.where(phi_idx_3 > len(phi_ax) - 1, phi_idx_2 % len(phi_ax), phi_idx_2)
            phi_idx_3 = np.where(phi_idx_3 > len(phi_ax) - 1, phi_idx_3 % len(phi_ax), phi_idx_3)
        else:
            print('case-3')
            # retain index for edge values
            phi_idx_0 = np.where(phi_idx_0 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_0)
            phi_idx_1 = np.where(phi_idx_1 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_1)
            phi_idx_2 = np.where(phi_idx_2 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_2)
            phi_idx_3 = np.where(phi_idx_3 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_3)

            phi_idx_0 = np.where(phi_idx_0 < 0, 0, phi_idx_0)
            phi_idx_1 = np.where(phi_idx_1 < 0, 0, phi_idx_1)
            phi_idx_2 = np.where(phi_idx_2 < 0, 0, phi_idx_2)
            phi_idx_3 = np.where(phi_idx_3 < 0, 0, phi_idx_3)

    # print('interpolating')

    if cubic:
        if flag == 0:
            # parallel interpolation
            for ii in prange(len(out_pattern)):
                p = np.zeros((4)).astype('float')
                temp = np.zeros((4)).astype('float')
                # 1 -  we interpolate along phi to find four theta points
                # output theta displacement within cell
                x_t = (theta_out[ii] - theta_idx_1[ii] * theta_step) / theta_step
                phi_idxs = np.zeros(4).astype(np.int64)
                phi_idxs[0] = phi_idx_0[ii]
                phi_idxs[1] = phi_idx_1[ii]
                phi_idxs[2] = phi_idx_2[ii]
                phi_idxs[3] = phi_idx_3[ii]
                # interpolating the 4 points along the output theta coordinate
                for jj in range(4):
                    # four points of the interpolation
                    p[0] = pattern[theta_idx_0[ii], phi_idxs[jj]]
                    p[1] = pattern[theta_idx_1[ii], phi_idxs[jj]]
                    p[2] = pattern[theta_idx_2[ii], phi_idxs[jj]]
                    p[3] = pattern[theta_idx_3[ii], phi_idxs[jj]]
                    # inner interpolation
                    temp[jj] = p[1] + 0.5 * x_t * (p[2] - p[0] + x_t * (
                            2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x_t * (
                            3.0 * (p[1] - p[2]) + p[3] - p[0])))
                # 2 -  finding the final point interpolating temp along phi
                # output phi displacement within cell
                x_p = (phi_out[ii] - phi_idx_1[ii] * phi_step) % (2 * np.pi) / phi_step
                point = np.real(temp[1] + 0.5 * x_p * (temp[2] - temp[0] + x_p * (
                        2.0 * temp[0] - 5.0 * temp[1] + 4.0 * temp[2] - temp[3] + x_p * (
                        3.0 * (temp[1] - temp[2]) + temp[3] - temp[0]))))
                out_pattern[ii] = point
                # if point <= 0:
                #     point = 0
        else:  # special version to handle negative theta_idx_0
            # parallel interpolation
            print('interpol', flag)
            for ii in prange(len(out_pattern)):
                p = np.zeros((4)).astype('float')
                temp = np.zeros((4)).astype('float')
                # 1 -  we interpolate along phi to find four theta points
                theta_idxs = np.zeros(4).astype(np.int64)
                theta_idxs[0] = theta_idx_0[ii]
                theta_idxs[1] = theta_idx_1[ii]
                theta_idxs[2] = theta_idx_2[ii]
                theta_idxs[3] = theta_idx_3[ii]

                # interpolating the 4 points along the output theta coordinate
                theta_sign = 0
                for jj in range(0, 4):
                    # save phi out value
                    phio = phi_out[ii]
                    # output phi displacement within cell
                    x_p = (phio - phi_idx_1[ii] * phi_step) % (2 * np.pi) / phi_step
                    # four points of the interpolation
                    p[0] = pattern[theta_idxs[jj], phi_idx_0[ii]]
                    p[1] = pattern[theta_idxs[jj], phi_idx_1[ii]]
                    p[2] = pattern[theta_idxs[jj], phi_idx_2[ii]]
                    p[3] = pattern[theta_idxs[jj], phi_idx_3[ii]]
                    if flag == 1:
                        # rotate phi idx if theta negative recall phi is odd last sample equal to first
                        if theta_idxs[jj] < 0:
                            # non permanent changes
                            phi_idxs_0 = int((phi_idx_0[ii] + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1))
                            phi_idxs_1 = int((phi_idx_1[ii] + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1))
                            phi_idxs_2 = int((phi_idx_2[ii] + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1))
                            phi_idxs_3 = int((phi_idx_3[ii] + (len(phi_ax) - 1) / 2) % (len(phi_ax) - 1))
                            # and make theta positive
                            theta_idxs[jj] = - theta_idxs[jj]
                            # and rotate out phi
                            phioi = (phio + np.pi) % (2 * np.pi)
                            theta_sign = 1
                            # displacement within cell
                            x_p = (phioi - phi_idxs_1 * phi_step) % (2 * np.pi) / phi_step
                            # four points of the interpolation
                            p[0] = pattern[theta_idxs[jj], phi_idxs_0]
                            p[1] = pattern[theta_idxs[jj], phi_idxs_1]
                            p[2] = pattern[theta_idxs[jj], phi_idxs_2]
                            p[3] = pattern[theta_idxs[jj], phi_idxs_3]
                    elif flag == 2:
                        # rotate phi idx if theta negative recall phi is even last sample before first
                        if theta_idxs[jj] < 0:
                            phi_idxs_0 = int((phi_idx_0[ii] + (len(phi_ax)) / 2) % (len(phi_ax)))
                            phi_idxs_1 = int((phi_idx_1[ii] + (len(phi_ax)) / 2) % (len(phi_ax)))
                            phi_idxs_2 = int((phi_idx_2[ii] + (len(phi_ax)) / 2) % (len(phi_ax)))
                            phi_idxs_3 = int((phi_idx_3[ii] + (len(phi_ax)) / 2) % (len(phi_ax)))
                            # and make theta positive
                            theta_idxs[jj] = - theta_idxs[jj]
                            # and rotate out phi
                            phioi = (phio + np.pi) % (2 * np.pi)
                            theta_sign = 1
                            # displacement within cell
                            x_p = (phioi - phi_idxs_1 * phi_step) % (2 * np.pi) / phi_step
                            # four points of the interpolation
                            p[0] = pattern[theta_idxs[jj], phi_idxs_0]
                            p[1] = pattern[theta_idxs[jj], phi_idxs_1]
                            p[2] = pattern[theta_idxs[jj], phi_idxs_2]
                            p[3] = pattern[theta_idxs[jj], phi_idxs_3]

                    # inner interpolation
                    temp[jj] = p[1] + 0.5 * x_p * (p[2] - p[0] + x_p * (
                            2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x_p * (
                            3.0 * (p[1] - p[2]) + p[3] - p[0])))
                # 2 -  finding the final point interpolating temp along theta
                # output phi displacement within cell
                x_t = (theta_out[ii] - theta_idx_1[ii] * theta_step) / theta_step
                point = np.real(temp[1] + 0.5 * x_t * (temp[2] - temp[0] + x_t * (
                        2.0 * temp[0] - 5.0 * temp[1] + 4.0 * temp[2] - temp[3] + x_t * (
                        3.0 * (temp[1] - temp[2]) + temp[3] - temp[0]))))
                out_pattern[ii] = point

    else:
        # parallel rect interpolation

        for ii in prange(len(out_pattern)):
            temp = np.zeros(2).astype('float')
            p = np.zeros(2).astype('float')
            # 1 -  we interpolate along phi to find 2 theta points
            # output theta displacement within cell
            x_t = (theta_out[ii] - theta_idx_1[ii] * theta_step) / theta_step
            phi_idxs = np.zeros(2).astype(np.int64)
            phi_idxs[0] = phi_idx_1[ii]
            phi_idxs[1] = phi_idx_2[ii]

            # interpolating the 4 points along the output theta coordinate
            for jj in range(2):
                # four points of the interpolation
                p[0] = pattern[theta_idx_1[ii], phi_idxs[jj]]
                p[1] = pattern[theta_idx_2[ii], phi_idxs[jj]]
                temp[jj] = p[0] + x_t * (p[1] - p[0])

            # 2 -  finding the final point interpolating temp along phi
            # output phi displacement within cell
            x_p = (phi_out[ii] - phi_idx_1[ii] * phi_step) / phi_step
            out_pattern[ii] = temp[0] + x_p * (temp[1] - temp[0])

    # print('returning')
    return out_pattern


if __name__ == '__main__':
    # NOTE: this is not going to work outside the simulator project, it requires an antenna object.
    from antenna import Antenna

    # this will load the pattern we already have
    antenna = Antenna(path='./Antenna_Pattern')

    theta = np.linspace(0, np.pi / 32, 1500)
    phi = np.linspace(0, np.pi * 2, 1200)
    Th, Ph = np.meshgrid(theta, phi)
    pattern_interp = 1j * np.ones_like(np.ravel(Th))
    pattern_interp = sphere_interp(np.ravel(Th.T), np.ravel(Ph.T), antenna.theta_ax, antenna.phi_ax,
                                   antenna.gain_matrix, pattern_interp, cubic=False)
    pattern_interp = pattern_interp.reshape(Th.T.shape)

    # %% display
    fig, ax = plt.subplots(1)
    ax.imshow(10 * np.log10(np.abs(antenna.gain_matrix)))

    fig, ax = plt.subplots(1)
    ax.imshow(10 * np.log10(np.abs(pattern_interp)))

    # %%
    fig, ax = plt.subplots(1)
    c = ax.pcolormesh(Th * np.cos(Ph), Th * np.sin(Ph), 10 * np.log10(np.abs(pattern_interp.T)),
                      cmap=plt.get_cmap('hot'))
    # c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
    c = ax.pcolormesh(Th * np.cos(Ph), Th * np.sin(Ph), (np.abs(pattern_interp.T)), cmap=plt.get_cmap('hot'))

    fig.colorbar(c)
    ax.set_xlabel("$\\theta\  cos \phi$")
    ax.set_ylabel("$\\theta\  sin \phi$")
