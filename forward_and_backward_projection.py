# Simone Mencarelli, October 2023
# This code is meant to test the forward and backward projection
# operators for the generalized array factor.
# This code simply aims to define the aperture to far field transformation and vice versa
# It is based on a square matrix to compute the array factor and its inverse.
# in order to get a larger number of points in the far field, the array factor matrix is repeated
# and the excitation coefficients set to zero outside of the actual array.

# %% imports
import numpy as np
import matplotlib.pyplot as plt
from arrayfactor import AntennaArray
from numpy import pi, sin, cos, tan
import matplotlib

# set pyqt5 rendering
matplotlib.use('Qt5Agg')

# %% define the array, i.e. a 2-d circular section array
# Reference uniform linear array
lam = 3e8 / 10e9
# element spacing
dx = lam / 2
# number of elements
N = 51
# x coordinates of the elements
x = np.arange(N) * dx - (N - 1) * dx / 2
# circular section angle
tc = 90 * pi / 180
# radius
r = x[-1] / (sin(tc / 2))
# individual points theta angle
t = np.linspace(-tc / 2, tc / 2, N)
# individual points x and y coordinates
xc = r * sin(t)
yc = r - r * cos(t)
# repeat the y coordinates and extend the xc coordinates arrays 11 times to get 11 times more points in the far field
xc = np.tile(xc, 11)
for i in range(11):
    if i != 5:
        xc[i * N:(i + 1) * N] += (i - 5) * (N * dx)
yc = np.tile(yc, 11)
# start with a uniform excitation for the array. the excitation has to be 0 outside the boundary of the original array
# create the excitation array
exc = np.zeros_like(xc)
# set the excitation to 1 for the central array
exc[N * 5: N * 6] = 1
# create array object
curved_array = AntennaArray(xc, yc, np.zeros_like(xc), exc, 10e9)

# plot the array excitation with circular markers
fig, ax = plt.subplots(1)
ax.scatter(curved_array.points[0], curved_array.points[1], marker='o', c=exc)
# axis labels
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_ylim(0, 0.1)
plt.show()

# %% The forward projection is the array factor
# define the angles for the far field, discretized in sin(theta) and phi
sintheta = np.linspace(-1, 1, len(xc)) # same length of the aperture to maintain the matrix square
phi = np.ones_like(sintheta) * 0
# compute the array factor
af = curved_array.factor(np.arcsin(sintheta), phi)
# plot the array factor
fig, ax = plt.subplots(1)
ax.plot(np.arcsin(sintheta), 20 * np.log10(np.abs(af)))
ax.set_xlabel('theta [rad]')
ax.set_ylabel('Array factor [dB]')
plt.show()

#%% For the backward projection we need to invert the transfer matrix of the array factor
# the transfer matrix is the matrix that relates the excitation coefficients to the far field
# the transfer matrix is a square matrix, so we can invert it
H = curved_array.H
# invert the transfer matrix
Hinv = np.linalg.inv(H)
# test the inversion
I = np.matmul(Hinv, H)
# return true if I is the identity matrix
print(np.allclose(I, np.eye(len(I))))
# compute the excitation coefficients from the array factor using the inverse transfer matrix
exc1 = np.matmul(Hinv, af)
# plot the excitation coefficients and compare to the original ones
fig, ax = plt.subplots(1)
ax.scatter(np.arange(len(exc1)), np.abs(exc1), marker='o', label='excitation coefficients from the array factor')
ax.scatter(np.arange(len(exc)), np.abs(exc), marker='x', label='original excitation coefficients')
ax.set_xlabel('Element number')
ax.set_ylabel('Excitation coefficient')
plt.show()

# %% compute the array factor of a uniform linear array and try to invert it to the curved array case
# Reference uniform linear array
exclin = np.ones_like(x)
uniform_array = AntennaArray(x, np.zeros_like(x), np.zeros_like(x), exclin, 10e9)
af_goal = uniform_array.factor(np.arcsin(sintheta), phi)


#%% forward projection

# try to find the curved array excitation coefficients using the back projection
exc2 = np.matmul(Hinv, af_goal)
# truncate the new excitation coefficients to be  outside of the original array
exc2[0: N * 5] = 0
exc2[N * 6:] = 0
# conservation of power
sigma = np.sqrt(np.sum(np.abs(exc2) ** 2) / N)
exc2 /= sigma
# plot the excitation coefficients and compare to the original ones
fig, ax = plt.subplots(1)
ax.scatter(np.arange(len(exc2)), np.abs(exc2), marker='o', label='excitation coefficients from the array factor')
ax.scatter(np.arange(len(exc)), np.abs(exc), marker='x', label='original excitation coefficients')
ax.set_xlabel('Element number')
ax.set_ylabel('Excitation coefficient')
plt.show()
# plot the phases
fig, ax = plt.subplots(1)
ax.scatter(np.arange(len(exc2)), np.angle(exc2), marker='o', label='excitation coefficients from the array factor')
ax.scatter(np.arange(len(exc)), np.angle(exc), marker='x', label='original excitation coefficients')
ax.set_xlabel('Element number')
ax.set_ylabel('Excitation coefficient phase [rad]')
plt.show()


#%% compute the array factor
curved_array.excitations = exc2
af2 = curved_array.factor(np.arcsin(sintheta), phi)
# plot the array factor and compare to the original one and the goal
fig, ax = plt.subplots(1)
ax.plot(np.arcsin(sintheta), 20 * np.log10(np.abs(af)), label='original array factor')
ax.plot(np.arcsin(sintheta), 20 * np.log10(np.abs(af_goal)), label='goal array factor')
ax.plot(np.arcsin(sintheta), 20 * np.log10(np.abs(af2)), label='new array factor')
ax.set_xlabel('theta [rad]')
ax.set_ylabel('Array factor [dB]')
ax.legend()
plt.show()



